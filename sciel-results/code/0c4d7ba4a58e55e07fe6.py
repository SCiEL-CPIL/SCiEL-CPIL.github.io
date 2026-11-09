def construct():
    import random
    import numpy as np

    N = 16
    M = N * N
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute distance squared matrix
    D = np.zeros((M, M), dtype=np.int32)
    for i in range(M):
        xi, yi = all_points[i]
        for j in range(i+1, M):
            xj, yj = all_points[j]
            d = (xi-xj)**2 + (yi-yj)**2
            D[i][j] = d
            D[j][i] = d

    best_size = M + 1
    best_set = None

    # Run multiple trials
    for trial in range(5000):
        random.seed(trial)
        S = []
        S_set = set()
        remaining = list(range(M))
        random.shuffle(remaining)
        
        for p in remaining:
            can_add = True
            n_s = len(S)
            # Check against all pairs in S
            for i_idx in range(n_s):
                i = S[i_idx]
                d_pi = D[p][i]
                for j_idx in range(i_idx + 1, n_s):
                    j = S[j_idx]
                    d_pj = D[p][j]
                    d_ij = D[i][j]
                    if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S.append(p)
                S_set.add(p)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
            if best_size <= 12:
                break

    if best_set is None:
        return all_points
        
    return [all_points[i] for i in best_set]
