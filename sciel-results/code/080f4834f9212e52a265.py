import numpy as np
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    
    # Precompute neighbors as bitsets
    neighbors_bits = [0] * N
    neighbors_list = [None] * N # Still need list for cover_count updates? Or just indices.
    
    for i in range(N):
        bits = 1 << i
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                j = i + (new_val - orig_val) * (q ** pos)
                bits |= (1 << j)
        neighbors_bits[i] = bits
        # Store list of neighbors for cover_count updates
        neighbors_list[i] = [i]
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                j = i + (new_val - orig_val) * (q ** pos)
                neighbors_list[i].append(j)

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 115

    all_mask = (1 << N) - 1

    while time.time() - start_time < time_limit:
        covered_bits = 0
        cover_count = np.zeros(N, dtype=int)
        C = []
        in_C_bits = 0

        # 1. Greedy Phase
        while covered_bits != all_mask:
            uncovered = (~covered_bits) & all_mask
            if uncovered == 0:
                break
            
            best_gain = -1
            best_idx = -1
            best_candidates = []
            
            # Find max gain
            # This loop is O(N) bit ops. 2187 * 1000 iters = 2M ops. Fast.
            for i in range(N):
                if (in_C_bits >> i) & 1:
                    continue
                gain = (neighbors_bits[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
                
            # Random tie-break
            best_idx = random.choice(best_candidates)
            
            C.append(best_idx)
            in_C_bits |= (1 << best_idx)
            covered_bits |= neighbors_bits[best_idx]
            
            for nb in neighbors_list[best_idx]:
                cover_count[nb] += 1
        
        if len(C) >= best_len:
            continue

        # 2. Local Search Phase
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant
            for i in range(len(C)):
                c = C[i]
                nb_list = neighbors_list[c]
                for nb in nb_list:
                    cover_count[nb] -= 1
                
                # Check if any nb has count 0
                redundant = False
                for nb in nb_list:
                    if cover_count[nb] == 0:
                        redundant = True
                        break
                
                if redundant:
                    for nb in nb_list:
                        cover_count[nb] += 1
                else:
                    C.pop(i)
                    in_C_bits &= ~(1 << c)
                    improved = True
                    break
            
            if not improved and C:
                # Step B: Swap
                best_swap = None
                best_delta = 0
                
                for c in C:
                    nb_c = neighbors_list[c]
                    # Find critical words
                    criticals = [nb for nb in nb_c if cover_count[nb] == 1]
                    
                    if not criticals:
                        continue
                    
                    # Find candidates that cover all criticals
                    # Start with neighbors of first critical
                    cand_bits = neighbors_bits[criticals[0]]
                    for w in criticals[1:]:
                        cand_bits &= neighbors_bits[w]
                        if cand_bits == 0:
                            break
                    
                    if cand_bits == 0:
                        continue
                    
                    # Exclude current C
                    cand_bits &= ~in_C_bits
                    
                    if cand_bits == 0:
                        continue
                    
                    # Iterate through set bits in cand_bits
                    # We want to minimize delta
                    # delta = (count of nb in only_c with count 2) - (count of nb in only_cp with count 1)
                    
                    # Precompute masks for c
                    mask_c = neighbors_bits[c]
                    # only_c = mask_c & ~mask_cp
                    # only_cp = mask_cp & ~mask_c
                    
                    # To speed up, we can just iterate
                    cp_iter = cand_bits
                    while cp_iter:
                        cp = (cp_iter & -cp_iter).bit_length() - 1
                        cp_iter &= cp_iter - 1
                        
                        mask_cp = neighbors_bits[cp]
                        
                        only_c = mask_c & ~mask_cp
                        only_cp = mask_cp & ~mask_c
                        
                        # Count
                        # We need to sum cover_count[only_c] == 2
                        # This requires iterating bits of only_c
                        # This is slow if done for every cp.
                        
                        # Optimization: Only check if cp is "promising"
                        # Heuristic: prefer cp with high gain?
                        
                        # For now, keep it simple but optimize the counting
                        # If only_c has many bits, it's expensive.
                        
                        # Let's skip this heavy optimization for now and rely on the bitset speedup for the candidate generation.
                        
                        count_2 = 0
                        oc_iter = only_c
                        while oc_iter:
                            idx = (oc_iter & -oc_iter).bit_length() - 1
                            oc_iter &= oc_iter - 1
                            if cover_count[idx] == 2:
                                count_2 += 1
                            
                        count_1 = 0
                        op_iter = only_cp
                        while op_iter:
                            idx = (op_iter & -op_iter).bit_length() - 1
                            op_iter &= op_iter - 1
                            if cover_count[idx] == 1:
                                count_1 += 1
                                
                        delta = count_2 - count_1
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (c, cp)
                
                if best_swap:
                    c, cp = best_swap
                    nb_c = neighbors_list[c]
                    nb_cp = neighbors_list[cp]
                    
                    for nb in nb_c:
                        cover_count[nb] -= 1
                    for nb in nb_cp:
                        cover_count[nb] += 1
                    
                    C.remove(c)
                    C.append(cp)
                    in_C_bits &= ~(1 << c)
                    in_C_bits |= (1 << cp)
                    improved = True
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    if best_C:
        # Final cleanup
        cover_count = np.zeros(N, dtype=int)
        for c in best_C:
            for nb in neighbors_list[c]:
                cover_count[nb] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_C)):
                c = best_C[i]
                nb_list = neighbors_list[c]
                for nb in nb_list:
                    cover_count[nb] -= 1
                redundant = False
                for nb in nb_list:
                    if cover_count[nb] == 0:
                        redundant = True
                        break
                if redundant:
                    for nb in nb_list:
                        cover_count[nb] += 1
                else:
                    best_C.pop(i)
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
