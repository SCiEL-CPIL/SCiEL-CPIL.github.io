import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_space = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]
    
    cover_bits = [0] * total_space
    if radius == 1:
        for i in range(total_space):
            bits = 1 << i
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                power = q ** (n - 1 - pos)
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * power
                    bits |= (1 << j)
            cover_bits[i] = bits
    else:
        for i in range(total_space):
            bits = 0
            wi = all_words[i]
            for j in range(total_space):
                wj = all_words[j]
                dist = sum(1 for a, b in zip(wi, wj) if a != b)
                if dist <= radius:
                    bits |= (1 << j)
            cover_bits[i] = bits

    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    for i in range(total_space):
        bits = cover_bits[i]
        while bits:
            lsb = bits & (-bits)
            idx = lsb.bit_length() - 1
            cover_matrix[i, idx] = True
            bits ^= lsb
            
    cm_int8 = cover_matrix.astype(np.int8)

    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        unc_int8 = uncovered.astype(np.int8)
        
        while np.any(uncovered):
            gains = cm_int8.dot(unc_int8)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = False
            unc_int8 = uncovered.astype(np.int8)
            
        return code

    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                min_count = np.min(cover_count[mask])
                
                if min_count > 1:
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    unique_mask = mask & (cover_count == 1)
                    
                    if np.any(unique_mask):
                        unique_indices = np.where(unique_mask)[0]
                        
                        covers_all = cover_matrix[:, unique_indices].all(axis=1)
                        
                        in_code = np.zeros(total_space, dtype=bool)
                        in_code[current_code] = True
                        
                        valid_mask = covers_all & (~in_code)
                        
                        if np.any(valid_mask):
                            valid_indices = np.where(valid_mask)[0]
                            new_idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                            
                            cover_count -= mask
                            cover_count += cover_matrix[new_idx]
                            current_code[i] = int(new_idx)
                            improved = True
                            break
                            
        return current_code

    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    best_size = len(best_code)
    
    # Run many restarts
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        
        if len(cand_code) < best_size:
            best_code = cand_code
            best_size = len(best_code)
            # Try to improve the new best further
            best_code = local_search_reduce(best_code)
            best_size = len(best_code)

    return [all_words[i] for i in best_code]
