import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    
    # Precompute neighbors
    neighbors = np.empty((N, sphere_size), dtype=int)
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

    # Precompute reverse neighbors (who covers word i?)
    rev_neighbors = [set() for _ in range(N)]
    for c in range(N):
        for nb in neighbors[c]:
            rev_neighbors[nb].add(c)

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def idx_to_word(idx):
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    while time.time() - start_time < time_limit:
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        C = []
        in_C = set()
        
        # 1. Greedy Phase
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            C.append(best_idx)
            in_C.add(best_idx)
            cover_count[neighbors[best_idx]] += 1
            covered[neighbors[best_idx]] = True
        
        if len(C) >= best_len:
            continue
            
        # 2. Local Search Phase
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant codewords
            for i in range(len(C)):
                c = C[i]
                nb = neighbors[c]
                if np.min(cover_count[nb]) > 1:
                    cover_count[nb] -= 1
                    C.pop(i)
                    in_C.remove(c)
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
                        continue
                    
                    # Find intersection of rev_neighbors of critical words
                    critical_list = list(critical_words)
                    critical_list.sort(key=lambda x: len(rev_neighbors[x]))
                    
                    candidates_set = set(rev_neighbors[critical_list[0]])
                    for w in critical_list[1:]:
                        candidates_set &= set(rev_neighbors[w])
                        if not candidates_set:
                            break
                    
                    if not candidates_set:
                        continue
                    
                    candidates_set -= in_C
                    
                    if not candidates_set:
                        continue
                    
                    for cp in candidates_set:
                        nb_cp = neighbors[cp]
                        
                        # Check validity: 
                        # Words in nb_c but not in nb_cp must have count >= 2
                        nb_cp_set = set(nb_cp)
                        valid = True
                        for v in nb_c:
                            if v not in nb_cp_set:
                                if cover_count[v] == 1:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        # Calculate delta
                        nb_c_set = set(nb_c)
                        only_cp = len(nb_cp_set - nb_c_set)
                        only_c = len(nb_c_set - nb_cp_set)
                        delta = only_cp - only_c
                        
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
                    in_C.remove(c)
                    in_C.add(cp)
                    improved = True
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    # Final optimization
    if best_C:
        in_C = set(best_C)
        cover_count = np.zeros(N, dtype=int)
        for c in best_C:
            cover_count[neighbors[c]] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_C)):
                c = best_C[i]
                nb = neighbors[c]
                if np.min(cover_count[nb]) > 1:
                    cover_count[nb] -= 1
                    best_C.pop(i)
                    in_C.remove(c)
                    improved = True
                    break
    
    result_words = []
    for idx in best_C:
        result_words.append(idx_to_word(idx))
        
    return result_words
