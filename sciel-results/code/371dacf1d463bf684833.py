import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # 1. Generate Candidates
    candidates = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # 2. Compute Conflict Matrix
    M = candidates @ candidates.T
    conflict = (M > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # 3. Multiple Greedy Initializations
    best_init_set = []
    best_init_size = 0
    rng = np.random.default_rng(42)
    
    for _ in range(20):
        order = rng.permutation(N)
        S = []
        in_S = np.zeros(N, dtype=bool)
        for idx in order:
            if time.time() > deadline - 100:
                break
            if len(S) == 0:
                S.append(idx)
                in_S[idx] = True
            else:
                if np.all(conflict[idx, in_S] == 0):
                    S.append(idx)
                    in_S[idx] = True
        if len(S) > best_init_size:
            best_init_size = len(S)
            best_init_set = S[:]
            
    # 4. Local Search
    best_set = best_init_set[:]
    best_mask = np.zeros(N, dtype=bool)
    for idx in best_set:
        best_mask[idx] = True
    
    def local_search(S, mask):
        """
        Attempts to improve S using Add and Swap 1->2.
        Returns improved S and mask.
        """
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1. Add
            if len(S) < N:
                if len(S) > 0:
                    compat_mask = np.all(conflict[:, mask] == 0, axis=1)
                    candidates_add = np.where(compat_mask & ~mask)[0]
                else:
                    candidates_add = np.where(~mask)[0]
                
                if len(candidates_add) > 0:
                    # Pick one with min degree in remaining graph?
                    # Just pick first for speed
                    v = candidates_add[0]
                    S.append(v)
                    mask[v] = True
                    improved = True
                    continue
            
            # 2. Swap 1->2
            # Try swapping each u in S
            # To avoid bias, shuffle the order of u
            S_list = S[:]
            rng.shuffle(S_list)
            
            for u in S_list:
                if time.time() > deadline - 10:
                    break
                
                # Temp remove u
                temp_mask = mask.copy()
                temp_mask[u] = False
                temp_S = [x for x in S if x != u]
                
                if len(temp_S) > 0:
                    compat_mask = np.all(conflict[:, temp_mask] == 0, axis=1)
                else:
                    compat_mask = np.ones(N, dtype=bool)
                
                candidates_add = np.where(compat_mask & ~mask)[0]
                
                if len(candidates_add) < 2:
                    continue
                
                # Find compatible pair
                rng.shuffle(candidates_add)
                found = False
                for i in range(len(candidates_add)):
                    v = candidates_add[i]
                    if i + 1 < len(candidates_add):
                        sub_conf = conflict[v, candidates_add[i+1:]]
                        compat_w = np.where(sub_conf == 0)[0]
                        if len(compat_w) > 0:
                            w = candidates_add[i+1 + compat_w[0]]
                            
                            # Swap
                            S.remove(u)
                            mask[u] = False
                            S.append(v)
                            mask[v] = True
                            S.append(w)
                            mask[w] = True
                            
                            improved = True
                            found = True
                            break
                            
                if found:
                    break
                    
        return S, mask

    # Run local search
    best_set, best_mask = local_search(best_set, best_mask)
    
    # If time remains, try to kick and re-search?
    # For now, just return.
    
    result = candidates[best_set].astype(np.int64)
    
    # Verification
    if len(result) == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    n = len(result)
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        vecs = []
        for i in range(D):
            v = [0]*D
            v[i] = 2
            vecs.append(v)
        return np.array(vecs, dtype=np.int64)
        
    return result
