# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points satisfying:
    - min pairwise squared distance >= max squared norm
    
    Uses a greedy approach: generate candidate integer vectors, then greedily
    select a maximal subset satisfying the distance constraint.
    
    Strategy: Use vectors of small norm (2, 3, 4) and greedily add points
    that maintain the constraint.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    d = 11
    
    # Generate candidate vectors with small norms
    # Norm 2: vectors with exactly two ±1 entries
    # Norm 3: vectors with exactly three ±1 entries  
    # Norm 4: vectors with one ±2, or four ±1s, or two ±1s and... 
    # Actually norm 4: (±2,0,...) or (±1,±1,±1,±1,0,...)
    
    candidates = []
    
    # Norm 2: C(11,2) * 4 = 220 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Norm 3: C(11,3) * 8 = 1320 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    # Norm 4: vectors with one ±2 entry: 11 * 2 = 22 vectors
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Norm 4: vectors with four ±1 entries: C(11,4) * 16 = 330 * 16 = 5280
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for l in range(k + 1, d):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute squared norms
    norms2 = np.sum(candidates ** 2, axis=1)
    
    # Sort by norm (prefer smaller norms first since they constrain less)
    order = np.argsort(norms2)
    candidates = candidates[order]
    norms2 = norms2[order]
    
    # Greedy selection: maintain that min pairwise dist^2 >= max norm^2 in selected set
    selected = []
    selected_norms2 = []
    
    for idx in range(n_cand):
        if time.time() > deadline:
            break
        
        v = candidates[idx]
        v_norm2 = norms2[idx]
        
        if len(selected) == 0:
            selected.append(v)
            selected_norms2.append(v_norm2)
            continue
        
        new_max_norm2 = max(v_norm2, max(selected_norms2))
        
        # Check if adding v maintains: min pairwise dist^2 >= new_max_norm2
        ok = True
        for sv in selected:
            diff = v - sv
            dist2 = np.sum(diff ** 2)
            if dist2 < new_max_norm2:
                ok = False
                break
        
        if ok:
            selected.append(v)
            selected_norms2.append(v_norm2)
    
    if len(selected) < 2:
        # Fallback to D11 minimal vectors
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
        return np.array(points, dtype=np.int64)
    
    result = np.array(selected, dtype=np.int64)
    result = np.around(result).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
