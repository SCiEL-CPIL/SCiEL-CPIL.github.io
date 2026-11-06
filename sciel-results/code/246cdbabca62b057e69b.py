"""Improved HP lattice folding with randomized growth and segment-relocation local search."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
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
            
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _try_relocate_segment(sequence, chain, start_idx, end_idx, occupied_by_others, rng):
    """
    Attempt to relocate the segment chain[start_idx:end_idx+1] to a new position.
    Returns the new segment path if successful, else None.
    """
    n = len(chain)
    seg_len = end_idx - start_idx + 1
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    # We want to find a path of length seg_len starting from a neighbor of left_neighbor
    # and ending at a neighbor of right_neighbor.
    # We perform a limited randomized walk.
    
    # Get candidate start positions (neighbors of left_neighbor)
    start_candidates = [p for p in _neighbours(left_neighbor) if p not in occupied_by_others]
    if not start_candidates:
        return None
    
    # Shuffle to randomize
    rng.shuffle(start_candidates)
    
    # We will try to grow a path from each start candidate
    for start_site in start_candidates:
        # BFS/DFS with limited depth to find a path to a neighbor of right_neighbor
        # Use a simple randomized walk with backtracking
        path = [start_site]
        visited = {start_site}
        
        # Try to grow towards right_neighbor
        for step in range(seg_len - 1):
            current = path[-1]
            options = []
            
            for site in _neighbours(current):
                if site in visited or site in occupied_by_others:
                    continue
                
                # Heuristic: prefer sites closer to right_neighbor
                dist_to_end = abs(site[0] - right_neighbor[0]) + abs(site[1] - right_neighbor[1])
                options.append((dist_to_end, rng.random(), site))
            
            if not options:
                break
            
            options.sort(key=lambda x: (x[0], x[1]))
            next_site = options[0][2]
            path.append(next_site)
            visited.add(next_site)
        
        # Check if we reached a neighbor of right_neighbor with the correct length
        if len(path) == seg_len:
            last = path[-1]
            if abs(last[0] - right_neighbor[0]) + abs(last[1] - right_neighbor[1]) == 1:
                return path
    
    return None

def _segment_move(sequence, chain, rng, max_attempts=200):
    """
    Local search via segment relocation moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_attempts):
        if _time_remaining() < 2.0:
            break
        
        # Pick a random segment of length 3 to 12 (or up to n-2)
        seg_len = min(rng.randint(3, 12), n - 2)
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        # Collect occupied positions excluding the segment
        occupied_by_others = {p: i for i, p in enumerate(chain) if i < start_idx or i > end_idx}
        
        # Try to relocate the segment
        new_segment = _try_relocate_segment(sequence, chain, start_idx, end_idx, occupied_by_others, rng)
        
        if new_segment is None:
            continue
        
        # Compute energy of the new chain
        new_chain = list(chain)
        for i, p in enumerate(new_segment):
            new_chain[start_idx + i] = p
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain

def fold(sequence, dimension):
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Randomized growth restarts
    # Spend about 65% of available time on growth
    growth_deadline = _time_remaining() * 0.65
    
    iterations = 0
    max_iterations = 15000
    
    while _time_remaining() > growth_deadline and iterations < max_iterations:
        iterations += 1
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search refinement on the best solution
    if _time_remaining() > 3.0:
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=800)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
