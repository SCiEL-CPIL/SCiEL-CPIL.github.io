import numpy as np
import itertools

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

    covered = np.zeros(N, dtype=bool)
    cover_count = np.zeros(N, dtype=int)
    C = []
    
    # 1. Greedy Phase
    while covered.sum() < N:
        gains = (~covered[neighbors]).sum(axis=1)
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        C.append(best_idx)
        cover_count[neighbors[best_idx]] += 1
        covered[neighbors[best_idx]] = True
        
    # 2. Local Search Phase
    in_C = np.zeros(N, dtype=bool)
    for c in C:
        in_C[c] = True
        
    improved = True
    while improved:
        improved = False
        
        # Step A: Remove redundant codewords
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
            # Step B: Swap to increase redundancy (minimize critical words)
            current_critical = np.sum(cover_count == 1)
            best_swap = None
            best_critical = current_critical
            
            for c in C:
                nb_c = neighbors[c]
                critical_words = nb_c[cover_count[nb_c] == 1]
                
                if len(critical_words) == 0:
                    continue
                
                if len(critical_words) == 1:
                    w = critical_words[0]
                    candidates = neighbors[w]
                else:
                    inter = set(neighbors[critical_words[0]].tolist())
                    for w in critical_words[1:]:
                        inter.intersection_update(neighbors[w].tolist())
                        if not inter:
                            break
                    if not inter:
                        continue
                    candidates = np.array(list(inter))
                
                mask = ~in_C[candidates]
                if not np.any(mask):
                    continue
                
                valid_candidates = candidates[mask]
                
                for cp in valid_candidates:
                    nb_cp = neighbors[cp]
                    temp_count = cover_count.copy()
                    temp_count[nb_c] -= 1
                    temp_count[nb_cp] += 1
                    
                    new_critical = np.sum(temp_count == 1)
                    
                    if new_critical < best_critical:
                        best_critical = new_critical
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

    result_words = []
    for idx in C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words
