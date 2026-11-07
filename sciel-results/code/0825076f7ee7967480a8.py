import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_MINV = P['minimum_vertices']

_ADJ = [0] * _N
for _a, _b in P['edges']:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)

def _calc_distinct_degrees(mask):
    degs = set()
    m = mask
    v = 0
    while m:
        if m & 1:
            degs.add((_ADJ[v] & mask).bit_count())
        m >>= 1
        v += 1
    return len(degs)

def _greedy_local_search(seed):
    rng = random.Random(seed)
    verts = list(range(_N))
    rng.shuffle(verts)
    
    # Initialize with _MINV random vertices
    current_set = set(verts[:_MINV])
    current_mask = 0
    for v in current_set:
        current_mask |= (1 << v)
    
    # Phase 1: Greedy Addition
    # Add vertices that increase distinct degree count
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_v = -1
        current_d = _calc_distinct_degrees(current_mask)
        
        for v in range(_N):
            if v in current_set:
                continue
            new_mask = current_mask | (1 << v)
            new_d = _calc_distinct_degrees(new_mask)
            gain = new_d - current_d
            if gain > best_gain:
                best_gain = gain
                best_v = v
        
        if best_v != -1 and best_gain > 0:
            current_set.add(best_v)
            current_mask |= (1 << best_v)
            improved = True

    # Phase 2: Local Search (Swaps)
    # Try to improve by swapping one vertex in with one vertex out
    improved = True
    while improved:
        improved = False
        current_d = _calc_distinct_degrees(current_mask)
        best_swap_gain = 0
        best_v_in = -1
        best_v_out = -1
        
        for v_in in range(_N):
            if v_in in current_set:
                continue
            for v_out in current_set:
                # Construct new mask
                new_mask = current_mask
                new_mask &= ~(1 << v_out)
                new_mask |= (1 << v_in)
                
                new_d = _calc_distinct_degrees(new_mask)
                gain = new_d - current_d
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_v_in = v_in
                    best_v_out = v_out
        
        if best_swap_gain > 0:
            current_set.remove(best_v_out)
            current_set.add(best_v_in)
            current_mask &= ~(1 << best_v_out)
            current_mask |= (1 << best_v_in)
            improved = True

    return current_mask, _calc_distinct_degrees(current_mask)

def construct():
    best_mask = 0
    best_d = 0
    
    # Run multiple random seeds to find the global optimum via heuristic
    # N=17 is small, so 1000 seeds is very fast
    for seed in range(2000):
        mask, d = _greedy_local_search(seed)
        if d > best_d:
            best_d = d
            best_mask = mask
    
    # Convert mask to list of vertices
    verts = []
    m = best_mask
    v = 0
    while m:
        if m & 1:
            verts.append(v)
        m >>= 1
        v += 1
        
    # Ensure minimum vertices
    if len(verts) < _MINV:
        for v in range(_N):
            if v not in verts:
                verts.append(v)
            if len(verts) >= _MINV:
                break
                
    return [int(v) for v in sorted(verts)]
