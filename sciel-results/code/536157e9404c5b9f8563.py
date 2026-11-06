"""Improved HP lattice folding with randomized growth and multi-attempt segment relocation."""
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
    return 105.0 - (time.time() - _START_TIME)

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
    Uses multiple random walk attempts per start candidate for robustness.
    Returns the new segment path if successful, else None.
    """
    n = len(chain)
    seg_len = end_idx - start_idx + 1
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    # Get candidate start positions (neighbors of left_neighbor)
    start_candidates = [p for p in _neighbours(left_neighbor) if p not in occupied_by_others]
    if not start_candidates:
        return None
    
    rng.shuffle(start_candidates)
    
    # For each start candidate, try multiple random walks
    max_walks_per_start = 15
    
    for start_pos in start_candidates:
        # Quick check: if segment length is 1, just verify adjacency
        if seg_len == 1:
            if abs(start_pos[0] - right_neighbor[0]) + abs(start_pos[1] - right_neighbor[1]) == 1:
                return [start_pos]
            continue
        
        # Try multiple random walks from this start position
        for _ in range(max_walks_per_start):
            path = [start_pos]
            visited = {start_pos}
            current = start_pos
            valid = True
            
            for step in range(1, seg_len):
                neighbors = [p for p in _neighbours(current) if p not in visited and p not in occupied_by_others]
                if not neighbors:
                    valid = False
                    break
                
                # Slight bias toward right_neighbor for longer segments
                if seg_len > 5 and step > seg_len // 2:
                    # Weight neighbors by inverse distance to right_neighbor
                    dists = [abs(p[0] - right_neighbor[0]) + abs(p[1] - right_neighbor[1]) for p in neighbors]
                    min_d = min(dists)
                    # Prefer closer neighbors but keep some randomness
                    weights = [1.0 / (d - min_d + 1) for d in dists]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    chosen = 0
                    for idx, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            chosen = idx
                            break
                    next_pos = neighbors[chosen]
                else:
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

def _segment_move(sequence, chain, rng, max_attempts=500, min_seg_len=3, max_seg_len=10):
    """
    Local search via segment relocation moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for attempt in range(max_attempts):
        if _time_remaining() < 3.0:
            break
        
        # Gradually reduce segment length as we progress (polish phase)
        progress = attempt / max_attempts
        if progress < 0.5:
            seg_min = min_seg_len
            seg_max = max_seg_len
        else:
            # Polish phase: shorter segments for fine-tuning
            seg_min = 2
            seg_max = max(3, max_seg_len - 3)
        
        seg_len = min(rng.randint(seg_min, seg_max), n - 2)
        if seg_len < 2:
            seg_len = 2
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
    
    if _time_remaining() < 2.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Randomized growth restarts
    # Spend about 60% of available time on growth
    growth_deadline = _time_remaining() * 0.60
    
    iterations = 0
    max_iterations = 20000
    
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
    if _time_remaining() > 5.0:
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=1000, min_seg_len=3, max_seg_len=12)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Final polish with very short segments if time allows
    if _time_remaining() > 3.0:
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=300, min_seg_len=2, max_seg_len=5)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
