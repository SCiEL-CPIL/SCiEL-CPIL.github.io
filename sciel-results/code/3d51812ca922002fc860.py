"""Improved HP lattice folding with randomized growth and segment-move local search."""
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
    Returns the new segment path (list of points) if successful, else None.
    Uses a randomized DFS to find a valid path of the required length.
    """
    seg_len = end_idx - start_idx + 1
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    # Candidate start positions: neighbors of left_neighbor that are free
    start_candidates = []
    for site in _neighbours(left_neighbor):
        if site not in occupied_by_others:
            start_candidates.append(site)
    
    if not start_candidates:
        return None
    
    # Candidate end positions: neighbors of right_neighbor that are free
    end_candidates = []
    for site in _neighbours(right_neighbor):
        if site not in occupied_by_others:
            end_candidates.append(site)
    
    if not end_candidates:
        return None
    
    # Randomized DFS to find a path of length seg_len from a start candidate to an end candidate
    # Limit the search depth to avoid excessive computation
    max_depth = 100
    
    for start_pos in start_candidates:
        # DFS with path tracking
        stack = [(start_pos, [start_pos])]
        visited = {start_pos}
        
        steps = 0
        while stack and steps < max_depth:
            steps += 1
            if _time_remaining() < 1.0:
                return None
            
            current, path = stack.pop()
            
            # Check if we've reached a valid end
            if len(path) == seg_len:
                if current in end_candidates:
                    return path
                continue
            
            # Try extending to neighbors
            for site in _neighbours(current):
                if site in visited or site in occupied_by_others:
                    continue
                if site == current:
                    continue
                visited.add(site)
                stack.append((site, path + [site]))
        
        if _time_remaining() < 1.0:
            return None
    
    return None

def _segment_move(sequence, chain, rng, max_attempts=200):
    """
    Local search via segment moves: pick a contiguous sub-chain and try to relocate it.
    A segment move preserves connectivity within the segment and to its neighbors.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_attempts):
        if _time_remaining() < 2.0:
            break
        
        # Pick a random segment of length 3 to 10 (or up to n-2)
        seg_len = min(rng.randint(3, 10), n - 2)
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        # The segment is chain[start_idx : end_idx+1]
        # Its neighbors in the chain are chain[start_idx-1] and chain[end_idx+1]
        
        # Collect occupied positions excluding the segment
        occupied_by_others = {p: i for i, p in enumerate(chain) if i < start_idx or i > end_idx}
        
        # Try to find a new location for the segment
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
    # Spend about 70% of available time on growth
    growth_deadline = _time_remaining() * 0.7
    
    iterations = 0
    max_iterations = 10000
    
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
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=300)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Second local search pass with fresh randomness
    if _time_remaining() > 5.0:
        rng2 = random.Random(99999 + len(sequence))
        best_chain = _segment_move(sequence, best_chain, rng2, max_attempts=300)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
