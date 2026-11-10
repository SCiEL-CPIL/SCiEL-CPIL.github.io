"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None

# Precompute neighbor offsets for speed
_NEIGHBOR_OFFSETS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

def _neighbours(point):
    x, y, z = point
    return ((x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1))

def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _compute_contacts(sequence, chain):
    """Return set of contact pairs (i, j) where i < j and they are H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts

def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        if rng.random() < 0.8:
            site = candidates[0][1]
        elif len(candidates) > 1:
            site = candidates[1][1]
        else:
            site = candidates[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _heuristic_grow(sequence, dimension, rng):
    """More sophisticated growth."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            potential = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q not in occupied:
                        potential += 0.5
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            trap_score = 0
            if free < 2:
                trap_score = 1.0
            
            score = gain * 2.0 + potential * 0.5 + 0.1 * free - trap_score + rng.random() * 0.3
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        k = min(3, len(candidates))
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            site = candidates[rng.randint(0, k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _pivot_move(sequence, chain, i, rng):
    """
    Perform a pivot move at index i.
    Rotates the suffix chain[i:] by 90 degrees around a random axis through chain[i-1].
    Returns (new_chain, energy_delta) or None if invalid.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    axis = rng.randint(0, 2)
    sign = rng.choice([-1, 1])
    
    new_suffix = []
    for p in suffix:
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        dz = p[2] - pivot[2]
        
        if axis == 0: # Rotate around X
            new_dx = dx
            new_dy = -sign * dz
            new_dz = sign * dy
        elif axis == 1: # Rotate around Y
            new_dx = sign * dz
            new_dy = dy
            new_dz = -sign * dx
        else: # Rotate around Z
            new_dx = -sign * dy
            new_dy = sign * dx
            new_dz = dz
            
        new_p = (pivot[0] + new_dx, pivot[1] + new_dy, pivot[2] + new_dz)
        new_suffix.append(new_p)
        
    # Check validity
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    prefix_set = set(chain[:i])
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    new_chain = chain[:i] + new_suffix
    
    # Compute energy delta efficiently
    # Only contacts involving indices >= i can change.
    # Let's just compute the new energy and compare? No, that's O(N).
    # Let's use the contact set approach or just recompute energy for the whole chain?
    # For N=136, recomputing energy is ~136*6 lookups. Very fast.
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    delta = old_energy - new_energy # positive if new is better (more negative)
    
    return new_chain, delta

def _segment_move(sequence, chain, rng):
    """
    Try to move a small segment.
    """
    n = len(chain)
    if n < 5:
        return None
    
    # Pick a random segment length
    seg_len = rng.choice([2, 3, 4])
    if n <= seg_len + 2:
        seg_len = max(2, n - 2)
        
    # Pick random start index
    i = rng.randint(1, n - seg_len - 1)
    j = i + seg_len - 1
    
    prev_pos = chain[i-1]
    next_pos = chain[j+1]
    
    # Get relative positions
    rel_pos = [(chain[k][0] - chain[i][0], chain[k][1] - chain[i][1], chain[k][2] - chain[i][2]) for k in range(i, j+1)]
    
    # Try all 6 neighbors of prev_pos
    old_first = chain[i]
    possible_new_firsts = []
    
    for site in _neighbours(prev_pos):
        if site == old_first:
            continue
        
        # Check if segment fits
        valid = True
        new_seg = []
        for k in range(seg_len):
            px, py, pz = site[0] + rel_pos[k][0], site[1] + rel_pos[k][1], site[2] + rel_pos[k][2]
            new_seg.append((px, py, pz))
            
            # Check collision with rest of chain (excluding the segment itself)
            # Indices involved in segment: i..j
            # We need to check if (px,py,pz) is in chain[0..i-1] or chain[j+1..n-1]
            # But we also need to ensure the last point connects to next_pos
            # Actually, easier to just build the full new chain and check self-avoidance?
            # For small segments, checking against the set of occupied positions is fast.
            # But we must exclude the positions of the segment itself from the "occupied" set for collision checks.
            # Let's just check against the global chain positions.
            
        # Check last point connection
        if new_seg[-1] != next_pos:
            diff = abs(new_seg[-1][0] - next_pos[0]) + abs(new_seg[-1][1] - next_pos[1]) + abs(new_seg[-1][2] - next_pos[2])
            if diff != 1:
                continue
                
        # Check collisions
        # Create set of occupied positions, removing the segment's old positions
        occupied = set(chain)
        for pos in chain[i:j+1]:
            occupied.discard(pos)
            
        for pos in new_seg:
            if pos in occupied:
                valid = False
                break
                
        if valid:
            possible_new_firsts.append(site)
            
    if not possible_new_firsts:
        return None
        
    new_first = rng.choice(possible_new_firsts)
    
    new_chain = list(chain)
    for k in range(seg_len):
        new_chain[i+k] = (new_first[0] + rel_pos[k][0], new_first[1] + rel_pos[k][1], new_first[2] + rel_pos[k][2])
        
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    delta = old_energy - new_energy
    
    return new_chain, delta

def _local_search(sequence, chain, rng, deadline):
    """
    Monte Carlo local search using pivot and segment moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temperature = 5.0
    cooling_rate = 0.99
    
    while time.time() < deadline and temperature > 0.1:
        move_type = rng.random()
        if move_type < 0.7:
            i = rng.randint(1, n - 1)
            result = _pivot_move(sequence, current_chain, i, rng)
        else:
            result = _segment_move(sequence, current_chain, rng)
            
        if result is not None:
            new_chain, delta = result
            # delta is old_energy - new_energy.
            # If delta > 0, new_energy is lower (better).
            # Energy change = -delta.
            # Accept if delta > 0 or with prob exp(delta / temperature)
            # Note: delta is positive for improvement.
            # Standard Metropolis: accept if dE < 0 or exp(-dE/T) > u.
            # Here dE = new_energy - old_energy = -delta.
            # So accept if -delta < 0 (delta > 0) or exp(delta/T) > u.
            
            if delta > 0:
                current_chain = new_chain
                current_energy -= delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            else:
                # delta <= 0
                if delta > -50: # Avoid underflow
                    if rng.random() < math.exp(delta / temperature):
                        current_chain = new_chain
                        current_energy -= delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
        
        temperature *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate budget for this sequence
    # We have 26 sequences. 330s total. ~12s per sequence.
    # Let's use a fraction of remaining time, capped.
    budget = min(remaining * 0.95, 12.0) 
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Generate multiple initial conformations
    # Use ~20% of budget for growth
    grow_deadline = time.time() + (deadline - time.time()) * 0.2
    n_grows = 0
    max_grows = 10
    
    while time.time() < grow_deadline and n_grows < max_grows:
        strategy = rng.random()
        if strategy < 0.4:
            chain = _greedy_grow(sequence, dimension, rng, temperature=1.0)
        elif strategy < 0.7:
            chain = _heuristic_grow(sequence, dimension, rng)
        else:
            chain = _greedy_grow(sequence, dimension, rng, temperature=2.0)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        n_grows += 1
        
    if best_chain is None:
        # Fallback: linear chain
        best_chain = [[i, 0, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search
    if time.time() < deadline - 0.5:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline - 0.5)
        
    return best_chain

# EVOLVE-BLOCK-END
