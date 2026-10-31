import numpy as np
import random
import time

def construct():
    def calc_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -np.inf
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        # FFT size
        size = 2 * max_u + 1
        n_fft = 1
        while n_fft < size:
            n_fft *= 2
            
        vec = np.zeros(n_fft, dtype=np.float64)
        for x in U:
            vec[x] = 1.0
            
        fft_vec = np.fft.fft(vec)
        
        # Sumset
        sum_poly = np.fft.ifft(fft_vec * fft_vec).real
        sum_counts = np.round(sum_poly).astype(int)
        size_sum = np.sum(sum_counts[:2*max_u+1] != 0)
        
        # Diffset
        corr = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
        diff_counts = np.round(corr).astype(int)
        
        size_diff = 0
        # Indices 0..max_u correspond to diffs 0..max_u
        # Indices n_fft-max_u..n_fft-1 correspond to diffs -max_u..-1
        # We need to count non-zeros in these ranges
        # Note: diff 0 is at index 0.
        # So we count indices 0 to max_u, and n_fft-max_u to n_fft-1.
        # But index 0 is included in both if we are not careful?
        # Range 1: [0, max_u]
        # Range 2: [n_fft - max_u, n_fft - 1]
        # These ranges are disjoint if max_u < n_fft/2, which is true since n_fft >= 2*max_u+1.
        size_diff += np.sum(diff_counts[0:max_u+1] != 0)
        size_diff += np.sum(diff_counts[n_fft-max_u:n_fft] != 0)
        
        if size_sum == 0 or size_diff == 0:
            return -np.inf
            
        ratio = size_diff / size_sum
        if ratio <= 0:
            return -np.inf
            
        score = 1 + np.log(ratio) / np.log(2 * max_u + 1)
        return score

    # Local search function
    def local_search(U, max_iters, time_limit, max_val):
        U = set(U)
        if 0 not in U:
            U.add(0)
        best_U = list(U)
        best_score = calc_score(best_U)
        
        start_time = time.time()
        iters = 0
        
        while iters < max_iters and (time.time() - start_time) < time_limit:
            U_curr = list(best_U)
            # Random mutation
            op = random.choice(['add', 'remove', 'move'])
            
            if op == 'add':
                if len(U_curr) < max_val: # Avoid huge sets?
                    new_val = random.randint(0, max_val)
                    if new_val not in U_curr:
                        U_curr.append(new_val)
            elif op == 'remove':
                if len(U_curr) > 1:
                    # Remove random non-zero
                    idx = random.randint(1, len(U_curr)-1)
                    U_curr.pop(idx)
            elif op == 'move':
                if len(U_curr) > 1:
                    idx = random.randint(1, len(U_curr)-1)
                    old_val = U_curr[idx]
                    new_val = random.randint(0, max_val)
                    if new_val not in U_curr:
                        U_curr[idx] = new_val
            
            if 0 not in U_curr:
                continue
                
            new_score = calc_score(U_curr)
            if new_score > best_score:
                best_score = new_score
                best_U = U_curr
            
            iters += 1
            
        return best_U, best_score

    # Multi-start search
    best_set_overall = [0, 1, 2, 6]
    best_score_overall = calc_score(best_set_overall)
    
    # Parameters for different runs
    # (target_size, max_val, time_per_run)
    # We have 120s total. Let's do 4 runs of 25s each, leaving 20s buffer.
    runs = [
        (50, 100, 25),
        (100, 200, 25),
        (200, 500, 25),
        (500, 1000, 25),
    ]
    
    random.seed(42)
    
    for target_size, max_val, time_limit in runs:
        # Initialize with random set
        U_init = set()
        U_init.add(0)
        while len(U_init) < target_size:
            U_init.add(random.randint(0, max_val))
        
        U_res, score_res = local_search(list(U_init), 10000, time_limit, max_val)
        
        if score_res > best_score_overall:
            best_score_overall = score_res
            best_set_overall = U_res
            
    return sorted(best_set_overall)
