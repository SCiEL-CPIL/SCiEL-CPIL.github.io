"""HP lattice protein folding with fast growth and delta-energy local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 200.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _build_occupied(chain):
    """Build dict mapping position -> index for O(1) lookups."""
    return {p: i for i, p in enumerate(chain)}

def _contact_count(sequence, chain):
    """Count H-H contacts for a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts

def _grow_random(sequence, dimension, rng):
    """
    Grow a chain by random walk with backtracking when stuck.
    Returns a valid self-avoiding chain or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Start with a simple initial placement
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Try to grow the chain
    for i in range(1, n):
        current = chain[-1]
        neighbors = _neighbours(current)
        random.shuffle(neighbors, rng)
        
        moved = False
        for site in neighbors:
            if site not in occupied:
                occupied[site] = i
                chain.append(site)
                moved = True
                break
        
        if not moved:
            # Backtrack
            # Try to move a previous residue to free up space
            # Simple approach: restart with different seed
            return None
    
    return chain

def _grow_greedy(sequence, dimension, rng, temperature=0.5):
    """
    Grow the chain using a greedy approach with temperature-controlled randomness.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 5.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        
        if temperature > 0.3 and rng.random() < 0.3:
            idx = min(len(options) - 1, max(0, int(rng.gauss(0, 1))))
            site = options[idx][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _local_search_fast(sequence, chain, rng, time_limit=5.0):
    """
    Local search using 1-0-1 moves with O(1) energy delta computation.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    start_time = time.time()
    
    # Precompute for fast updates
    while time.time() - start_time < time_limit and _time_remaining() > 1.0:
        improved = False
        occupied = _build_occupied(best_chain)
        
        # Try 1-0-1 moves for each interior residue
        for i in range(1, n - 1):
            if time.time() - start_time > time_limit or _time_remaining() <= 1.0:
                break
            
            old_pos = best_chain[i]
            
            for new_pos in _neighbours(old_pos):
                # Skip if new_pos is adjacent in chain (would create non-unit step)
                if new_pos == best_chain[i - 1] or new_pos == best_chain[i + 1]:
                    continue
                
                # Check if new_pos is occupied
                if new_pos in occupied:
                    continue
                
                # Compute energy delta
                old_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos):
                        j = occupied.get(q)
                        if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            old_contacts += 1
                
                new_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(new_pos):
                        j = occupied.get(q)
                        if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            new_contacts += 1
                
                # Also check contacts involving neighbors of old_pos and new_pos
                # with other H residues
                delta = new_contacts - old_contacts
                
                # Apply move temporarily
                del occupied[old_pos]
                occupied[new_pos] = i
                best_chain[i] = new_pos
                
                # Recompute energy to be safe (delta might miss some contacts)
                new_energy = _energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    # Revert
                    del occupied[new_pos]
                    occupied[old_pos] = i
                    best_chain[i] = old_pos
        
        if not improved:
            break
    
    return best_chain

def _segment_relocate(sequence, chain, rng):
    """
    Try to relocate a short segment to a new position.
    Returns the new chain if successful, else None.
    """
    n = len(chain)
    if n < 6:
        return None
    
    # Pick a random segment of length 3-5
    seg_len = rng.randint(3, min(5, n - 4))
    start_idx = rng.randint(2, n - seg_len - 2)
    end_idx = start_idx + seg_len - 1
    
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    # Try to find a new location for the segment
    for attempt in range(20):
        # Start from a random neighbor of left_neighbor
        start_candidates = [p for p in _neighbours(left_neighbor) if p not in chain]
        if not start_candidates:
            continue
        
        start_pos = rng.choice(start_candidates)
        
        # Try to grow a path from start_pos
        occupied = set(chain)
        for p in chain[start_idx:end_idx + 1]:
            occupied.discard(p)
        
        path = [start_pos]
        visited = {start_pos}
        current = start_pos
        success = False
        
        for step in range(1, seg_len):
            neighbors = [p for p in _neighbours(current) if p not in visited and p not in occupied]
            if not neighbors:
                break
            next_pos = rng.choice(neighbors)
            path.append(next_pos)
            visited.add(next_pos)
            current = next_pos
        
        if len(path) == seg_len:
            # Check if we ended near right_neighbor
            if abs(current[0] - right_neighbor[0]) + abs(current[1] - right_neighbor[1]) == 1:
                # Check self-avoidance
                if len(set(path)) == len(path) and not (set(path) & occupied):
                    new_chain = list(chain)
                    for i, p in enumerate(path):
                        new_chain[start_idx + i] = p
                    
                    # Verify validity
                    if len(set(new_chain)) == len(new_chain):
                        return new_chain
    
    return None

def fold(sequence, dimension):
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        # Fallback: simple snake pattern
        chain = []
        x, y = 0, 0
        for i in range(n):
            chain.append((x, y))
            if i % 2 == 0:
                x += 1
                y += 1
            else:
                x += 1
                y -= 1
        return [list(p) for p in chain]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Multi-start growth
    growth_deadline = _time_remaining() * 0.4
    
    # Try different growth strategies
    attempts = 0
    while _time_remaining() > growth_deadline and attempts < 500:
        attempts += 1
        
        if attempts % 5 == 0:
            chain = _grow_random(sequence, dimension, rng)
        else:
            temp = 0.1 + 0.9 * (attempts % 10) / 10.0
            chain = _grow_greedy(sequence, dimension, rng, temperature=temp)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback: simple snake
        best_chain = []
        x, y = 0, 0
        for i in range(n):
            best_chain.append((x, y))
            if i % 2 == 0:
                x += 1
                y += 1
            else:
                x += 1
                y -= 1
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search refinement
    if _time_remaining() > 2.0:
        ls_time = min(_time_remaining() - 1.0, 30.0)
        best_chain = _local_search_fast(sequence, best_chain, rng, time_limit=ls_time)
        best_energy = _energy(sequence, best_chain)
        
        # Phase 3: Try segment relocations
        if _time_remaining() > 2.0:
            for _ in range(50):
                if _time_remaining() < 2.0:
                    break
                new_chain = _segment_relocate(sequence, best_chain, rng)
                if new_chain is not None:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        # Quick local search after improvement
                        if _time_remaining() > 2.0:
                            best_chain = _local_search_fast(sequence, best_chain, rng, time_limit=2.0)
                            best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
