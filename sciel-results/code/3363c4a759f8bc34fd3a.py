"""
HP lattice protein folding on the cubic lattice.
Strategy: Contact-targeted backtracking with constraint propagation.
We pre-compute the target number of H-H contacts from the ground state energy,
then use a CSP-style backtracking search to find a self-avoiding walk
that realizes this specific set of contacts.
"""
import random
import time
import math
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin before 330s kill

def _neighbours_list(point):
    """Return list of 6 face-neighbours of point in 3D."""
    x, y, z = point
    return [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]

def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours_list(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _count_contacts(sequence, chain):
    """Count non-adjacent H-H contacts."""
    return -_energy(sequence, chain)

def _greedy_grow(sequence, dimension, rng, bias=1.0):
    """
    Fast initial self-avoiding walk with heuristic bias towards H contacts.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours_list(last)
        
        options = []
        for site in neighbors:
            if site in occupied:
                continue
            
            site_neighbors = _neighbours_list(site)
            h_score = 0
            free_score = 0
            
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            score = h_score * bias + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def _backtrack_fold(sequence, target_contacts, time_budget, rng):
    """
    Backtracking search with constraint propagation to find a self-avoiding walk
    that achieves exactly target_contacts H-H contacts.
    
    We grow the chain from left to right, at each step choosing a position
    that maximizes the potential to achieve the target contact count.
    """
    n = len(sequence)
    deadline = time.time() + time_budget
    
    # We'll use a DFS with pruning based on achievable contacts
    # State: current chain prefix, current contact count, max possible remaining contacts
    
    best_chain = None
    best_energy = float('inf')
    
    # Start with a few different greedy seeds
    seeds = []
    for _ in range(5):
        chain = _greedy_grow(sequence, 3, rng, bias=1.0 + rng.random())
        if chain:
            e = _energy(sequence, chain)
            seeds.append((chain, e))
    
    if not seeds:
        # Fallback: straight line
        return [(i, 0, 0) for i in range(n)]
    
    # Sort seeds by energy (most negative first)
    seeds.sort(key=lambda x: x[1])
    
    # For each seed, do a local search using backtracking on the suffix
    for seed_chain, seed_energy in seeds:
        if time.time() > deadline:
            break
        
        # Try to improve this seed using backtracking on the last k residues
        # We'll fix the first n-k residues and regrow the last k
        # Start with small k and increase
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try different suffix lengths
            max_k = min(n, int(time_budget * 5))  # Cap based on time
            k = min(max_k, n - 2)  # Need at least 2 residues in prefix
            
            for attempt in range(3):
                if time.time() > deadline:
                    break
                
                # Pick a random k
                k = rng.randint(2, max_k)
                prefix_len = n - k
                prefix = seed_chain[:prefix_len]
                
                # Regrow the suffix
                new_suffix = _regrow_suffix(sequence, prefix, k, target_contacts, seed_energy, time_budget/10, rng)
                
                if new_suffix:
                    new_chain = prefix + new_suffix
                    new_energy = _energy(sequence, new_chain)
                    
                    if new_energy < seed_energy:
                        seed_chain = new_chain
                        seed_energy = new_energy
                        improved = True
                        break
                    
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = new_chain
            
            if best_chain is None:
                best_chain = seed_chain
                best_energy = seed_energy
    
    if best_chain is None:
        best_chain = seeds[0][0]
        best_energy = seeds[0][1]
    
    return best_chain

def _regrow_suffix(sequence, prefix, k, target_contacts, current_energy, time_budget, rng):
    """
    Regrow the last k residues of the chain, trying to achieve target_contacts.
    Uses backtracking with constraint propagation.
    """
    n = len(sequence)
    prefix_len = len(prefix)
    deadline = time.time() + time_budget
    
    # Count contacts in the prefix
    prefix_contacts = 0
    prefix_occupied = {p: i for i, p in enumerate(prefix)}
    for i, p in enumerate(prefix):
        if sequence[i] == 'H':
            for q in _neighbours_list(p):
                j = prefix_occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    prefix_contacts += 1
    
    # We need (target_contacts - prefix_contacts) more contacts from the suffix
    needed_contacts = max(0, target_contacts - prefix_contacts)
    
    # Backtracking DFS
    def dfs(pos, chain, occupied, contacts):
        nonlocal deadline
        
        if time.time() > deadline:
            return None
        
        if pos == n:
            # Check if we achieved the target
            if contacts >= needed_contacts:
                return chain[:]
            return None
        
        last = chain[-1]
        
        # Try all possible next positions
        neighbors = _neighbours_list(last)
        rng.shuffle(neighbors)
        
        for next_pos in neighbors:
            if next_pos in occupied:
                continue
            
            # Check if this creates a contact
            new_contact = 0
            if sequence[pos] == 'H':
                for sn in _neighbours_list(next_pos):
                    j = occupied.get(sn)
                    if j is not None and j < pos - 1 and sequence[j] == 'H':
                        new_contact += 1
            
            new_chain = chain + [next_pos]
            new_occupied = dict(occupied)
            new_occupied[next_pos] = pos
            
            result = dfs(pos + 1, new_chain, new_occupied, contacts + new_contact)
            if result:
                return result
            
            # If we've exceeded the needed contacts, prune
            if contacts + new_contact > needed_contacts + 2:
                continue
        
        return None
    
    # Start DFS from the end of the prefix
    result = dfs(prefix_len, prefix[:], dict(prefix_occupied), prefix_contacts)
    
    if result:
        return result[prefix_len:]
    
    return None

def _local_search(sequence, chain, time_budget, rng):
    """
    Local search with simulated annealing.
    """
    n = len(chain)
    start_time = time.time()
    
    current_chain = chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    # SA parameters
    T0 = max(1.0, abs(best_energy) * 0.5)
    T_final = 0.1
    
    while time.time() < start_time + time_budget:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move
            result = _pivot_move(sequence, current_chain, rng)
        else:
            # Slide move
            result = _slide_move(sequence, current_chain, rng)
        
        if result is None:
            continue
        
        new_chain, delta = result
        
        # Metropolis acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T0):
            current_chain = new_chain
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
        
        # Cool down
        T0 *= 0.999
    
    return best_chain, best_energy

def _pivot_move(sequence, chain, rng):
    """
    Pivot move: pick a random split point, rotate the suffix by 90 degrees around one axis.
    """
    n = len(chain)
    if n < 3:
        return None
    
    split = rng.randint(1, n - 2)
    axis = rng.randint(0, 2)
    direction = rng.choice([1, -1])
    
    def rotate(point, axis, direction):
        x, y, z = point
        if axis == 0:
            if direction == 1:
                return (x, -z, y)
            else:
                return (x, z, -y)
        elif axis == 1:
            if direction == 1:
                return (z, y, -x)
            else:
                return (-z, y, x)
        else:
            if direction == 1:
                return (y, -x, z)
            else:
                return (-y, x, z)
    
    prefix = chain[:split]
    suffix = chain[split:]
    pivot = chain[split-1]
    
    new_suffix = []
    for p in suffix:
        tx = p[0] - pivot[0]
        ty = p[1] - pivot[1]
        tz = p[2] - pivot[2]
        rx, ry, rz = rotate((tx, ty, tz), axis, direction)
        new_suffix.append((rx + pivot[0], ry + pivot[1], rz + pivot[2]))
    
    new_chain = prefix + new_suffix
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    # Check unit steps
    for i in range(n - 1):
        p1 = new_chain[i]
        p2 = new_chain[i+1]
        dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
        if dist != 1:
            return None
    
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    delta = new_energy - old_energy
    
    return (new_chain, delta)

def _slide_move(sequence, chain, rng):
    """
    Slide move: pick a random interior residue and try to move it to an adjacent empty site.
    """
    n = len(chain)
    if n < 3:
        return None
    
    idx = rng.randint(1, n - 2)
    pos = chain[idx]
    neighbors = _neighbours_list(pos)
    rng.shuffle(neighbors)
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    for new_pos in neighbors:
        if new_pos in occupied:
            continue
        
        prev_pos = chain[idx - 1]
        next_pos = chain[idx + 1]
        
        d1 = abs(prev_pos[0]-new_pos[0]) + abs(prev_pos[1]-new_pos[1]) + abs(prev_pos[2]-new_pos[2])
        d2 = abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2])
        
        if d1 != 1 or d2 != 1:
            continue
        
        new_chain = chain[:]
        new_chain[idx] = new_pos
        
        if len(set(new_chain)) != n:
            continue
        
        old_energy = _energy(sequence, chain)
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - old_energy
        
        return (new_chain, delta)
    
    return None

def fold(sequence, dimension):
    """
    Best conformation using Contact-Targeted Backtracking + Local Search.
    """
    n = len(sequence)
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 1.0:
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time per sequence
    budget = 10.0
    if remaining - budget < 5.0:
        budget = remaining - 5.0
    
    if budget < 1.0:
        budget = 0.5
    
    rng = random.Random(12345 + len(sequence))
    
    # Estimate target contacts: use a heuristic based on the sequence
    # For benchmark chains, the ground state energy is known to be negative
    # We'll aim for a reasonable number of contacts
    h_count = sequence.count('H')
    # Rough estimate: about 30-50% of H-H pairs can be in contact
    target_contacts = max(1, int(h_count * (h_count - 1) / 2 * 0.3))
    
    # 1. Try backtracking search
    best_chain = _backtrack_fold(sequence, target_contacts, budget * 0.6, rng)
    best_energy = _energy(sequence, best_chain)
    
    # 2. Refine with local search
    final_chain, final_energy = _local_search(sequence, best_chain, budget * 0.4, rng)
    
    if final_energy < best_energy:
        best_chain = final_chain
        best_energy = final_energy
    
    # Final check: ensure self-avoidance
    if len(set(best_chain)) != n:
        return [[i, 0, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
