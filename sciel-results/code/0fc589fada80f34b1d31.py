import itertools
import time
import random
import numpy as np

N = 16

def _precompute_conflicts():
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    pts = np.array(points)
    
    # Pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    
    # conflict[a, b] is a bitmask of p such that (a, b, p) is isosceles
    # We want a V x V matrix of integers.
    # conflict[a, b] has bit p set if dists[a,b]==dists[a,p] or dists[a,b]==dists[b,p] or dists[a,p]==dists[b,p]
    
    # Initialize conflict matrix as 0
    conflict = np.zeros((V, V), dtype=np.int64) # 256 bits fit in int64? 2^64 is huge. 256 bits needs 4 uint64s or Python int.
    # Python int is better for arbitrary size, but NumPy int64 might be faster for bitwise ops if we stay within 64 bits?
    # N=16 -> 256 points. 256 bits > 64 bits.
    # So we need either Python ints or an array of uint64s.
    # Python ints are fine for 256 bits.
    
    conflict_py = [[0] * V for _ in range(V)]
    
    # Vectorized check for each pair (a, b)
    # For a fixed a, b:
    # mask1 = (dists[a, :] == dists[a, b]) # p such that d(a,p) == d(a,b)
    # mask2 = (dists[b, :] == dists[a, b]) # p such that d(b,p) == d(a,b)
    # mask3 = (dists[a, :] == dists[b, :]) # p such that d(a,p) == d(b,p)
    # conflict_p = mask1 | mask2 | mask3
    # Convert boolean mask to integer bitmask
    
    for a in range(V):
        for b in range(a + 1, V):
            d_ab = dists[a, b]
            # Points p where d(a,p) == d_ab
            m1 = (dists[a, :] == d_ab)
            # Points p where d(b,p) == d_ab
            m2 = (dists[b, :] == d_ab)
            # Points p where d(a,p) == d(b,p)
            m3 = (dists[a, :] == dists[b, :])
            
            combined = m1 | m2 | m3
            
            # Convert numpy boolean array to int bitmask
            # np.packbits? Or just loop?
            # Loop is 256 iterations. 256*256/2 = 32768 pairs. 32768 * 256 = 8M ops. Fine.
            mask_val = 0
            for p in range(V):
                if combined[p]:
                    mask_val |= (1 << p)
            
            conflict_py[a][b] = mask_val
            conflict_py[b][a] = mask_val
            
    return points, conflict_py

def _greedy_bitmask(points, conflict, seed):
    random.seed(seed)
    V = len(points)
    order = list(range(V))
    random.shuffle(order)
    
    chosen = set()
    chosen_mask = 0
    blocked_mask = 0
    
    for p in order:
        if (chosen_mask | blocked_mask) & (1 << p):
            continue
        
        # Check if we can add p
        # If we reach here, p is not chosen and not blocked.
        # Add p
        chosen.add(p)
        chosen_mask |= (1 << p)
        
        # Update blocked_mask
        # For every q in chosen (excluding p), p and q form a pair.
        # Any point r in conflict[p][q] is now blocked.
        for q in chosen:
            if q == p:
                continue
            blocked_mask |= conflict[p][q]
            
    return [points[i] for i in sorted(chosen)]

def construct():
    start_time = time.time()
    points, conflict = _precompute_conflicts()
    
    best_set = None
    best_size = float('inf')
    
    deadline = start_time + 290 # 10s margin
    
    seed = 0
    while time.time() < deadline:
        result = _greedy_bitmask(points, conflict, seed)
        if result:
            size = len(result)
            if size < best_size:
                best_size = size
                best_set = result
        seed += 1
        
    if best_set is None:
        # Fallback
        best_set = [points[0]]
        
    return best_set
