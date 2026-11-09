import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0
    
    # 1. Precompute APs
    ap_list = []
    idx_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for start in range(n):
            indices = []
            curr = start
            while curr < n:
                indices.append(curr)
                curr += d
            if len(indices) > 0:
                ap_id = len(ap_list)
                ap_list.append(indices)
                for idx in indices:
                    idx_to_aps[idx].append(ap_id)
    
    num_aps = len(ap_list)
    offset = n
    size = 2 * n + 1
    
    # 2. Greedy Construction
    signs = [0] * n
    sums = [0] * num_aps
    counts = [0] * size
    counts[0] = num_aps
    current_max = 0
    rng = random.Random(42)
    
    def calc_new_max(affected, delta, cur_sums, cur_counts, cur_max):
        # Simulate applying delta to affected APs
        # Returns new_max
        
        max_new_val = 0
        for ap_id in affected:
            new_s = cur_sums[ap_id] + delta
            abs_new = abs(new_s)
            if abs_new > max_new_val:
                max_new_val = abs_new
        
        if max_new_val > cur_max:
            return max_new_val
            
        # Check if cur_max bucket is emptied
        count_cur_max_before = 0
        count_cur_max_after = 0
        
        for ap_id in affected:
            old_s = cur_sums[ap_id]
            new_s = old_s + delta
            
            if abs(old_s) == cur_max:
                count_cur_max_before += 1
            if abs(new_s) == cur_max:
                count_cur_max_after += 1
        
        total_cur_max = cur_counts[cur_max + offset] + cur_counts[-cur_max + offset]
        new_total_cur_max = total_cur_max - count_cur_max_before + count_cur_max_after
        
        if new_total_cur_max > 0:
            return cur_max
        else:
            # Max dropped. Find next max.
            for d in range(cur_max - 1, -1, -1):
                virtual_count = cur_counts[d + offset] + cur_counts[-d + offset]
                for ap_id in affected:
                    old_s = cur_sums[ap_id]
                    new_s = old_s + delta
                    if abs(old_s) == d:
                        virtual_count -= 1
                    if abs(new_s) == d:
                        virtual_count += 1
                if virtual_count > 0:
                    return d
            return 0

    for i in range(n):
        affected = idx_to_aps[i]
        
        max_plus = calc_new_max(affected, 1, sums, counts, current_max)
        max_minus = calc_new_max(affected, -1, sums, counts, current_max)
        
        if max_plus <= max_minus:
            chosen_sign = 1
            new_max = max_plus
        else:
            chosen_sign = -1
            new_max = max_minus
            
        signs[i] = chosen_sign
        delta = chosen_sign
        
        for ap_id in affected:
            old_s = sums[ap_id]
            new_s = old_s + delta
            sums[ap_id] = new_s
            counts[old_s + offset] -= 1
            counts[new_s + offset] += 1
            
        current_max = new_max

    best_signs = signs[:]
    best_max = current_max
    best_sums = sums[:]
    best_counts = counts[:]

    # 3. Iterated Hill Climbing
    opt_signs = signs[:]
    opt_sums = sums[:]
    opt_counts = counts[:]
    opt_max = current_max

    def get_max_disc(c):
        for d_val in range(n, -1, -1):
            if c[d_val + offset] > 0 or c[-d_val + offset] > 0:
                return d_val
        return 0

    while time.time() - start_time < time_limit:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            i = rng.randint(0, n - 1)
            old_sign = opt_signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = idx_to_aps[i]
            
            for ap_id in affected:
                old_s = opt_sums[ap_id]
                new_s = old_s + delta
                opt_sums[ap_id] = new_s
                opt_counts[old_s + offset] -= 1
                opt_counts[new_s + offset] += 1
            
            new_max = opt_max
            if opt_counts[opt_max + offset] == 0 and opt_counts[-opt_max + offset] == 0:
                while new_max > 0 and opt_counts[new_max + offset] == 0 and opt_counts[-new_max + offset] == 0:
                    new_max -= 1
            for ap_id in affected:
                s_val = abs(opt_sums[ap_id])
                if s_val > new_max:
                    new_max = s_val
            
            if new_max < opt_max:
                opt_max = new_max
                improved = True
                if opt_max < best_max:
                    best_max = opt_max
                    best_signs = opt_signs[:]
                    best_sums = opt_sums[:]
                    best_counts = opt_counts[:]
            else:
                for ap_id in affected:
                    new_s = opt_sums[ap_id]
                    old_s = new_s - delta
                    opt_sums[ap_id] = old_s
                    opt_counts[new_s + offset] -= 1
                    opt_counts[old_s + offset] += 1
        
        k = rng.randint(5, 15)
        for _ in range(k):
            i = rng.randint(0, n - 1)
            opt_signs[i] *= -1
            delta = 2 * opt_signs[i]
            for ap_id in idx_to_aps[i]:
                old_s = opt_sums[ap_id]
                new_s = old_s + delta
                opt_sums[ap_id] = new_s
                opt_counts[old_s + offset] -= 1
                opt_counts[new_s + offset] += 1
        
        opt_max = get_max_disc(opt_counts)

    return best_signs
