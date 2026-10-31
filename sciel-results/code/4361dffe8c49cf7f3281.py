import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_vectors = []
    best_score = 0
    
    # Seed: k=2
    # Generate all k=2 vectors
    vecs_k2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vecs_k2.append(v)
    best_vectors = vecs_k2
    best_score = len(vecs_k2)
    
    # Try k=3
    if time.time() < deadline:
        vecs_k3 = []
        for comb in itertools.combinations(range(D), 3):
            for signs in itertools.product([1, -1], repeat=3):
                v = [0] * D
                for idx, val in zip(comb, signs):
                    v[idx] = val
                vecs_k3.append(v)
        
        # Greedy Independent Set for k=3
        # Conflict: share 2 coords with same signs
        # We will try a few random orders
        
        best_k3 = []
        n = len(vecs_k3)
        # Precompute a fast conflict check?
        # Or just check on the fly.
        # n=1320. O(n^2) is 1.7M checks. Fast.
        
        for restart in range(50):
            if time.time() > deadline: break
            indices = list(range(n))
            np.random.shuffle(indices)
            # Heuristic: prioritize vectors with fewer potential conflicts? 
            # Or just random.
            
            current_set = []
            # To speed up, we can maintain a set of "blocked" vectors?
            # Or just check against current_set.
            # Current set size might be ~200.
            # 1320 * 200 * 3 (coords) = 800k ops. Very fast.
            
            for idx in indices:
                v = vecs_k3[idx]
                ok = True
                for u in current_set:
                    # Check if u, v conflict
                    # Count common indices with same sign
                    common_same = 0
                    for k in range(D):
                        if v[k] != 0 and u[k] == v[k]:
                            common_same += 1
                    if common_same >= 2:
                        ok = False
                        break
                if ok:
                    current_set.append(v)
            
            if len(current_set) > len(best_k3):
                best_k3 = current_set
                
        if len(best_k3) > best_score:
            best_vectors = best_k3
            best_score = len(best_k3)

    # Try k=4
    if time.time() < deadline:
        vecs_k4 = []
        for comb in itertools.combinations(range(D), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for idx, val in zip(comb, signs):
                    v[idx] = val
                vecs_k4.append(v)
        
        best_k4 = []
        n = len(vecs_k4)
        
        for restart in range(20): # Fewer restarts, larger space
            if time.time() > deadline: break
            indices = list(range(n))
            np.random.shuffle(indices)
            
            current_set = []
            for idx in indices:
                v = vecs_k4[idx]
                ok = True
                for u in current_set:
                    common_same = 0
                    for k in range(D):
                        if v[k] != 0 and u[k] == v[k]:
                            common_same += 1
                    # Conflict if u.v > R/2 = 2.
                    # u.v = sum(u_k v_k).
                    # If common_same >= 3, dot >= 3. Conflict.
                    # If common_same == 2, dot could be 2, 0, -2.
                    # Wait, if common_same == 2, and the other common coords have opposite signs?
                    # We need exact dot product.
                    dot = 0
                    for k in range(D):
                        if v[k] != 0:
                            dot += v[k] * u[k]
                    if dot > 2:
                        ok = False
                        break
                if ok:
                    current_set.append(v)
            
            if len(current_set) > len(best_k4):
                best_k4 = current_set
                
        if len(best_k4) > best_score:
            best_vectors = best_k4
            best_score = len(best_k4)

    return np.array(best_vectors, dtype=np.int64)
