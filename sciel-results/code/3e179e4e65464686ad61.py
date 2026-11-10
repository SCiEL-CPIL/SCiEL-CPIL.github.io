import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    # Field operations
    k = 3
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
        
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    # Encode point (xa, xb, ya, yb) to index
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute line_id[point_idx, dir_idx]
    # dir_idx 0..48: m = (ma, mb), ma = dir_idx // p, mb = dir_idx % p
    # dir_idx 49: vertical
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = encode(xa, xb, ya, yb)
                    x = (xa, xb)
                    y = (ya, yb)
                    for ma in range(p):
                        for mb in range(p):
                            d_idx = ma * p + mb
                            m = (ma, mb)
                            c = field_add(y, field_neg(field_mul(m, x)))
                            line_id[idx, d_idx] = c[0] * p + c[1]
                    # Vertical direction
                    line_id[idx, n_dirs - 1] = xa * p + xb

    # Helper to check validity of S (boolean array)
    # Returns True if S is a valid complement (N is valid Nikodym set)
    def is_valid(S):
        # line_count[d, l] = number of points in S on line l in direction d
        line_count = np.zeros((n_dirs, q), dtype=np.int32)
        
        # Accumulate counts
        # For each direction d, bincount the line_ids of points in S
        s_indices = np.where(S)[0]
        if len(s_indices) == 0:
            return True # S empty is valid (N all points)
            
        for d in range(n_dirs):
            lids = line_id[s_indices, d]
            line_count[d] = np.bincount(lids, minlength=q)
            
        # Check condition for each point
        # counts_for_points[x, d] = line_count[d, line_id[x, d]]
        # We can build this matrix
        counts = np.empty((n_points, n_dirs), dtype=np.int32)
        for d in range(n_dirs):
            counts[:, d] = line_count[d, line_id[:, d]]
            
        # Condition:
        # If S[x] is True, we need counts[x, d] == 1 for some d
        # If S[x] is False, we need counts[x, d] == 0 for some d
        cond_S = S[:, None] & (counts == 1)
        cond_notS = (~S)[:, None] & (counts == 0)
        valid_dir = cond_S | cond_notS
        satisfied = np.any(valid_dir, axis=1)
        
        return np.all(satisfied)

    # Fast local search
    start_time = time.time()
    deadline = start_time + 590
    
    best_S = np.zeros(n_points, dtype=bool)
    best_size = 0
    
    # Precompute list of all points
    all_points = list(range(n_points))
    
    # Try multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Initialize S
        # Strategy: Start with empty S and greedily add
        # Or start with random small S
        current_S = np.zeros(n_points, dtype=bool)
        
        # Greedy phase
        # Randomize order of points to add
        order = all_points[:]
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            current_S[idx] = True
            if is_valid(current_S):
                pass # Keep
            else:
                current_S[idx] = False
                
        current_size = np.sum(current_S)
        
        # Local search phase
        # Try to add more points
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try all points not in S
            candidates = [i for i in range(n_points) if not current_S[i]]
            random.shuffle(candidates)
            for idx in candidates:
                if time.time() > deadline:
                    break
                current_S[idx] = True
                if is_valid(current_S):
                    changed = True
                    current_size += 1
                    # Update best if better
                    if current_size > best_size:
                        best_S = current_S.copy()
                        best_size = current_size
                else:
                    current_S[idx] = False
                    
        # Try swaps if stuck
        # This is expensive, so do it sparingly
        if time.time() < deadline - 10:
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Pick random u in S, v not in S
                s_indices = np.where(current_S)[0]
                ns_indices = np.where(~current_S)[0]
                if len(s_indices) == 0 or len(ns_indices) == 0:
                    break
                u = s_indices[random.randint(0, len(s_indices)-1)]
                v = ns_indices[random.randint(0, len(ns_indices)-1)]
                
                current_S[u] = False
                current_S[v] = True
                if is_valid(current_S):
                    # Accept swap (size same, but might open doors)
                    # Don't update best size, but keep state
                    pass
                else:
                    current_S[u] = True
                    current_S[v] = False

    # Convert best_S to N
    N_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = encode(xa, xb, ya, yb)
                    if not best_S[idx]:
                        N_points.append([[xa, xb], [ya, yb]])
                        
    if not N_points:
        # Fallback
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
