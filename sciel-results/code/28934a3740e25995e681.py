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
    neighbors_bitset = [0] * N
    
    # Precompute neighbors
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        neighbors_bitset[i] |= (1 << i)
        idx += 1
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                j = i + (new_val - orig_val) * (q ** pos)
                neighbors[i, idx] = j
                neighbors_bitset[i] |= (1 << j)
                idx += 1

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    while time.time() - start_time < time_limit:
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        C = []
        
        # 1. Greedy Phase
        while covered.sum() < N:
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
                # Step B: Swap using bitsets
                best_swap = None
                best_delta = 0 # We want to minimize "cost" or maximize "freedom"
                
                # To make it faster, we can just look for ANY valid swap that doesn't increase size,
                # but we want to find one that leads to a smaller set later.
                # A good heuristic: Prefer swaps where the new codeword covers many words with count > 1 (increasing redundancy)
                # or simply find a valid swap and break.
                
                for c in C:
                    nb_c = neighbors[c]
                    # Find words covered only by c
                    unique_mask = (cover_count == 1)
                    # We need words in nb_c that are unique to c
                    # unique_words = nb_c[cover_count[nb_c] == 1]
                    unique_words = nb_c[cover_count[nb_c] == 1]
                    
                    if len(unique_words) == 0:
                        # This shouldn't happen if Step A ran, but just in case
                        continue
                    
                    # Build bitset for unique_words
                    unique_bitset = 0
                    for w in unique_words:
                        unique_bitset |= (1 << w)
                    
                    # Find c' not in C such that N(c') covers unique_bitset
                    # (neighbors_bitset[c'] & unique_bitset) == unique_bitset
                    found = False
                    for cp in range(N):
                        if in_C[cp]:
                            continue
                        if (neighbors_bitset[cp] & unique_bitset) == unique_bitset:
                            # Valid swap candidate
                            # Heuristic: Prefer cp that covers more redundant words (count > 1)
                            # Or just take the first one?
                            # Taking the first one might get stuck.
                            # Let's try to pick the one that maximizes "new redundancy"
                            # new_redundancy = sum(1 for w in N(cp) if cover_count[w] > 1)
                            # This is expensive to compute for every cp.
                            # Let's just take the first valid one to ensure progress.
                            best_swap = (c, cp)
                            found = True
                            break
                    
                    if found:
                        break
                
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
