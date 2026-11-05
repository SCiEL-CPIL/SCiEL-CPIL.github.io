import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    neighbors = np.empty((N, sphere_size), dtype=int)
    
    # Precompute neighbors
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        idx += 1
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                idx += 1

    # Precompute adjacency matrix M_adj: M_adj[cp, w] = 1 if w is in N(cp)
    # This is the transpose of the "word to neighbors" relation.
    M_adj = np.zeros((N, N), dtype=bool)
    for cp in range(N):
        M_adj[cp, neighbors[cp]] = True

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for output conversion
    
    while time.time() - start_time < time_limit:
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        C = []
        
        # 1. Greedy Phase with Randomization
        while covered.sum() < N:
            # Calculate gains: number of new words covered by each candidate
            # gain[i] = number of neighbors of i that are not yet covered
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
                
            C.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            covered[neighbors[best_idx]] = True
        
        if len(C) >= best_len:
            continue
            
        # 2. Local Search Phase
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
            
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant codewords
            # We can vectorize this check?
            # A codeword c is redundant if all its neighbors are covered by other codewords.
            # i.e., for all w in N(c), cover_count[w] > 1.
            # This is equivalent to: min(cover_count[N(c)]) > 1.
            
            # We iterate through C to remove one by one to update counts.
            for i in range(len(C)):
                c = C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
            
            if not improved and C:
                # Step B: Swap
                best_swap = None
                best_delta = 0
                
                for c in C:
                    nb_c = neighbors[c]
                    critical_mask = (cover_count[nb_c] == 1)
                    critical_words = nb_c[critical_mask]
                    
                    if len(critical_words) == 0:
                        # Should have been caught by removal, but just in case
                        continue
                    
                    # Find candidates cp such that N(cp) covers all critical_words
                    # candidates_mask[cp] is True if cp covers all critical_words
                    candidates_mask = np.ones(N, dtype=bool)
                    for w in critical_words:
                        candidates_mask &= M_adj[:, w]
                        if not candidates_mask.any():
                            break
                            
                    if not candidates_mask.any():
                        continue
                    
                    # Exclude current codewords
                    candidates_mask[in_C] = False
                    
                    if not candidates_mask.any():
                        continue
                        
                    valid_candidates = np.where(candidates_mask)[0]
                    
                    # Evaluate swaps
                    for cp in valid_candidates:
                        # Check validity: no word in N(c) \ N(cp) has count 1
                        # only_c = N(c) \ N(cp)
                        only_c_mask = M_adj[c] & ~M_adj[cp]
                        
                        if np.any(cover_count[only_c_mask] == 1):
                            continue
                        
                        # Calculate delta (heuristic for stability)
                        # We want to increase redundancy (count==2) and decrease fragility (count==1)
                        only_cp_mask = M_adj[cp] & ~M_adj[c]
                        
                        delta = np.sum(cover_count[only_c_mask] == 2) - np.sum(cover_count[only_cp_mask] == 1)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (c, cp)
                
                if best_swap:
                    c, cp = best_swap
                    nb_c = neighbors[c]
                    nb_cp = neighbors[cp]
                    
                    cover_count[nb_c] -= 1
                    cover_count[nb_cp] += 1
                    
                    C.remove(c)
                    C.append(cp)
                    in_C[c] = False
                    in_C[cp] = True
                    improved = True
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    # Final optimization: try to remove more codewords
    if best_C:
        in_C = np.zeros(N, dtype=bool)
        for c in best_C:
            in_C[c] = True
        cover_count = np.zeros(N, dtype=int)
        for c in best_C:
            cover_count[neighbors[c]] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_C)):
                c = best_C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    best_C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
    
    result_words = []
    for idx in best_C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words
