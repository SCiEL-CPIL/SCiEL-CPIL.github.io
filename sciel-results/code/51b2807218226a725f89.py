import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    total_words = q ** n
    
    def int_to_base_q(num, n, q):
        digits = []
        for _ in range(n):
            digits.append(num % q)
            num //= q
        return digits
    
    words_repr = [int_to_base_q(i, n, q) for i in range(total_words)]
    
    balls = [0] * total_words
    for i in range(total_words):
        mask = 0
        wi = words_repr[i]
        for j in range(total_words):
            wj = words_repr[j]
            dist = 0
            for k in range(n):
                if wi[k] != wj[k]:
                    dist += 1
                    if dist > radius:
                        break
            if dist <= radius:
                mask |= (1 << j)
        balls[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Greedy
    uncovered = full_mask
    code = []
    
    while uncovered and time.time() < deadline - 20:
        best_c = -1
        best_count = 0
        for c in range(total_words):
            cnt = (balls[c] & uncovered).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_c = c
        
        if best_c == -1 or best_count == 0:
            break
            
        code.append(best_c)
        uncovered &= ~balls[best_c]
        
    def perform_reduction(code):
        if not code:
            return code
        
        cover_counts = np.zeros(total_words, dtype=np.int8)
        for c in code:
            mask = balls[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                mask ^= lsb
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                mask = balls[c]
                redundant = True
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    mask ^= lsb
                
                if redundant:
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        mask ^= lsb
                    
                    code.pop(i)
                    changed = True
        
        return code

    code = perform_reduction(code)
    
    if time.time() < deadline - 5:
        cover_counts = np.zeros(total_words, dtype=np.int8)
        for c in code:
            mask = balls[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                mask ^= lsb
        
        uncovered_count = np.sum(cover_counts == 0)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        best_code = list(code)
        best_energy = len(code) * 1000 + uncovered_count
        
        all_words = list(range(total_words))
        
        while T > T_min and time.time() < deadline - 2:
            move_type = random.random()
            
            delta_energy = 0
            
            if move_type < 0.3:
                if not code:
                    T *= cooling
                    continue
                idx = random.randint(0, len(code) - 1)
                c = code[idx]
                
                mask = balls[c]
                new_uncovered_increase = 0
                while mask:
                    lsb = mask & -mask
                    idx_w = lsb.bit_length() - 1
                    if cover_counts[idx_w] == 1:
                        new_uncovered_increase += 1
                    mask ^= lsb
                
                delta_energy = -1 * 1000 + new_uncovered_increase
                
                if random.random() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code.pop(idx)
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx_w = lsb.bit_length() - 1
                        cover_counts[idx_w] -= 1
                        mask ^= lsb
                    
                    uncovered_count += new_uncovered_increase
                    
            elif move_type < 0.6:
                c = random.choice(all_words)
                if c in code:
                    T *= cooling
                    continue
                    
                mask = balls[c]
                new_uncovered_decrease = 0
                while mask:
                    lsb = mask & -mask
                    idx_w = lsb.bit_length() - 1
                    if cover_counts[idx_w] == 0:
                        new_uncovered_decrease += 1
                    mask ^= lsb
                
                delta_energy = 1 * 1000 - new_uncovered_decrease
                
                if random.random() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code.append(c)
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx_w = lsb.bit_length() - 1
                        cover_counts[idx_w] += 1
                        mask ^= lsb
                    
                    uncovered_count -= new_uncovered_decrease
                    
            else:
                if not code:
                    T *= cooling
                    continue
                idx = random.randint(0, len(code) - 1)
                c_old = code[idx]
                c_new = random.choice(all_words)
                if c_new == c_old or c_new in code:
                    T *= cooling
                    continue
                
                mask_old = balls[c_old]
                mask_new = balls[c_new]
                
                delta_uncovered = 0
                
                mask_only_old = mask_old & ~mask_new
                while mask_only_old:
                    lsb = mask_only_old & -mask_only_old
                    idx_w = lsb.bit_length() - 1
                    if cover_counts[idx_w] == 1:
                        delta_uncovered += 1
                    mask_only_old ^= lsb
                
                mask_only_new = mask_new & ~mask_old
                while mask_only_new:
                    lsb = mask_only_new & -mask_only_new
                    idx_w = lsb.bit_length() - 1
                    if cover_counts[idx_w] == 0:
                        delta_uncovered -= 1
                    mask_only_new ^= lsb
                
                delta_energy = delta_uncovered
                
                if random.random() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code[idx] = c_new
                    
                    mask = mask_old
                    while mask:
                        lsb = mask & -mask
                        idx_w = lsb.bit_length() - 1
                        cover_counts[idx_w] -= 1
                        mask ^= lsb
                        
                    mask = mask_new
                    while mask:
                        lsb = mask & -mask
                        idx_w = lsb.bit_length() - 1
                        cover_counts[idx_w] += 1
                        mask ^= lsb
                        
                    uncovered_count += delta_uncovered

            current_energy = len(code) * 1000 + uncovered_count
            if current_energy < best_energy:
                best_energy = current_energy
                best_code = list(code)
                
            T *= cooling
            
            if random.random() < 0.01:
                code = perform_reduction(code)
                cover_counts = np.zeros(total_words, dtype=np.int8)
                for c in code:
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx_w = lsb.bit_length() - 1
                        cover_counts[idx_w] += 1
                        mask ^= lsb
                uncovered_count = np.sum(cover_counts == 0)

        code = perform_reduction(best_code)
        
    return [words_repr[c] for c in code]
