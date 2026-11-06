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
    Returns the new segment path if successful, else None.
    """
    n = len(chain)
    seg_len = end_idx - start_idx + 1
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    # We want to find a path of length seg_len starting from a neighbor of left_neighbor
    # and ending at a neighbor of right_neighbor.
    # We perform a limited DFS/BFS or randomized walk.
    
    # Get candidate start positions (neighbors of left_neighbor)
    start_candidates = [p for p in _neighbours(left_neighbor) if p not in occupied_by_others]
    if not start_candidates:
        return None
    
    # Shuffle to randomize
    rng.shuffle(start_candidates)
    
    # We will try to grow a path from each start candidate
    # To keep it fast, we limit the depth or just do a simple greedy/randomized walk.
    # Given the small segment lengths (3-10), a simple randomized walk with backtracking
    # or just a few random tries is sufficient.
    
    # For efficiency, we just try a few random walks from the best start candidates.
    # A "random walk" here means picking random valid neighbors.
    
    # To improve success rate, we can bias towards the right_neighbor.
    # But simple random walk with self-avoidance against occupied_by_others and current path is key.
    
    for start_pos in start_candidates:
        current = start_pos
        path = [current]
        visited = {current}
        
        # If the segment is length 1, we just check if start_pos is adjacent to right_neighbor
        if seg_len == 1:
            if abs(current[0] - right_neighbor[0]) + abs(current[1] - right_neighbor[1]) == 1:
                return path
            continue
        
        success = False
        # Try a limited number of random extensions
        # Since seg_len is small, we can try a few random paths.
        for _ in range(20): # Limit attempts per start pos to keep it fast
            path = [start_pos]
            visited = {start_pos}
            current = start_pos
            
            valid = True
            for step in range(1, seg_len):
                # Get neighbors of current not in visited and not in occupied_by_others
                neighbors = [p for p in _neighbours(current) if p not in visited and p not in occupied_by_others]
                if not neighbors:
                    valid = False
                    break
                
                # Choose a neighbor
                # Bias towards right_neighbor if close? 
                # For now, random choice.
                next_pos = rng.choice(neighbors)
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
            
            if not valid:
                continue
            
            # Check if we ended at a position adjacent to right_neighbor
            if abs(current[0] - right_neighbor[0]) + abs(current[1] - right_neighbor[1]) == 1:
                return path
        
    return None

def _segment_move(sequence, chain, rng, max_attempts=200):
    """
    Local search via segment relocation.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_attempts):
        if _time_remaining() < 2.0:
            break
        
        # Pick a random segment
        # Longer segments are harder to relocate but offer more energy change potential.
        # Shorter segments (3-5) are easier to relocate.
        seg_len = rng.randint(3, min(8, n - 2))
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        # Occupied by others (excluding the segment itself)
        occupied_by_others = {p: i for i, p in enumerate(chain) if i < start_idx or i > end_idx}
        
        new_segment = _try_relocate_segment(sequence, chain, start_idx, end_idx, occupied_by_others, rng)
        
        if new_segment is None:
            continue
        
        # Construct new chain
        new_chain = list(chain)
        for i, p in enumerate(new_segment):
            new_chain[start_idx + i] = p
        
        # Verify self-avoidance just in case (should be guaranteed by logic, but safe)
        # And compute energy
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
        # Increase attempts for local search as relocation is more effective
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=1000)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
