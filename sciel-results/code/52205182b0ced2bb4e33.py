import math
import time
import random
import numpy as np

def construct():
    START_TIME = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s kill
    
    def elapsed():
        return time.time() - START_TIME
    
    def remaining():
        return DEADLINE - elapsed() - 3  # 3s safety margin
    
    def compute_score(u):
        """Compute the score for a set u. Uses NumPy for efficiency."""
        u = np.array(sorted(set(u)), dtype=np.int64)
        if len(u) == 0 or u[0] != 0:
            return -1.0
        max_u = u[-1]
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        
        n = len(u)
        
        # For n up to ~200, direct NumPy computation is very fast
        if n <= 200:
            sums = u[:, None] + u[None, :]
            diffs = u[:, None] - u[None, :]
            sumset_size = len(np.unique(sums))
            diffset_size = len(np.unique(diffs))
        else:
            # FFT-based approach for very large sets (unlikely to be reached in 120s with this search)
            max_val = max_u
            n_fft = 1
            while n_fft < 2 * (2 * max_val + 1):
                n_fft *= 2
            
            indicator = np.zeros(n_fft, dtype=np.float64)
            indicator[u] = 1.0
            
            ft = np.fft.fft(indicator)
            sum_conv = np.fft.ifft(ft * ft).real
            sumset_size = np.sum(np.round(sum_conv) > 0.5)
            
            diff_conv = np.fft.ifft(ft * np.conj(ft)).real
            half = n_fft // 2
            positive_lags = np.sum(np.round(diff_conv[:half+1]) > 0.5)
            negative_lags = np.sum(np.round(diff_conv[half+1:]) > 0.5)
            diffset_size = positive_lags + negative_lags
        
        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        """Check if u is a Golomb ruler using NumPy."""
        u = np.array(sorted(u), dtype=np.int64)
        n = len(u)
        if n < 2:
            return True
        diffs = np.abs(u[:, None] - u[None, :])
        # We need to check if all n*(n-1)/2 pairwise differences are distinct
        # Since diffs is symmetric and has 0 on diagonal, we can just flatten and unique
        # But unique on full matrix is O(n^2 log n^2).
        # Better: take upper triangle
        idx = np.triu_indices(n, 1)
        d = diffs[idx]
        return len(np.unique(d)) == len(d)
    
    # Known good Golomb rulers (optimal or near-optimal)
    known_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 40],
        [0, 1, 4, 10, 18, 23, 29, 37, 47, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 47, 58, 64],
        [0, 1, 4, 10, 18, 23, 29, 37, 47, 58, 64, 75],
    ]
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    for u in known_rulers:
        if not is_golomb(u):
            continue
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
    
    def local_search(base_u, time_limit):
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        
        improved = True
        iteration = 0
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            iteration += 1
            current_max = max(best_u)
            n = len(best_u)
            search_hi = current_max + max(20, n * 4)
            search_lo = 1
            
            indices_to_try = list(range(1, n))
            random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if (time.time() - start_time) > time_limit:
                    break
                old_val = best_u[idx]
                window = max(10, n * 4)
                lo = max(search_lo, old_val - window)
                hi = min(search_hi, old_val + window)
                
                for new_val in range(lo, hi + 1):
                    if new_val == old_val:
                        continue
                    if new_val in best_u:
                        continue
                    
                    candidate = sorted([x if x != old_val else new_val for x in best_u])
                    
                    if not is_golomb(candidate):
                        continue
                    
                    s = compute_score(candidate)
                    if s > best_score_local + 1e-9:
                        best_score_local = s
                        best_u = candidate
                        improved = True
                        break  # Restart with new best
        
        return best_u, best_score_local
    
    def grow_search(base_u, time_limit):
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            current_max = max(best_u)
            n = len(best_u)
            
            search_hi = current_max + max(20, n * 5)
            for new_val in range(1, search_hi + 1):
                if (time.time() - start_time) > time_limit:
                    break
                if new_val in best_u:
                    continue
                
                candidate = sorted(best_u + [new_val])
                if not is_golomb(candidate):
                    continue
                
                s = compute_score(candidate)
                if s > best_score_local + 1e-9:
                    best_score_local = s
                    best_u = candidate
                    improved = True
                    break
        
        return best_u, best_score_local
    
    def greedy_construct(start_u, time_limit):
        start_time = time.time()
        best_u = sorted(start_u)
        best_score_local = compute_score(best_u)
        
        while (time.time() - start_time) < time_limit:
            n = len(best_u)
            current_max = max(best_u)
            search_hi = current_max + max(30, n * 6)
            best_new_val = None
            best_new_score = best_score_local
            
            for new_val in range(1, search_hi + 1):
                if (time.time() - start_time) > time_limit:
                    break
                if new_val in best_u:
                    continue
                
                candidate = sorted(best_u + [new_val])
                if not is_golomb(candidate):
                    continue
                
                s = compute_score(candidate)
                if s > best_new_score:
                    best_new_score = s
                    best_new_val = new_val
            
            if best_new_val is None:
                break
            
            best_u = sorted(best_u + [best_new_val])
            best_score_local = best_new_score
        
        return best_u, best_score_local

    if remaining() > 30:
        t0a = min(remaining() - 25, 25)
        g_u, g_score = greedy_construct([0, 1, 4, 10], t0a)
        if g_score > best_score:
            best_set = g_u
            best_score = g_score
    
    if remaining() > 25:
        t0b = min(remaining() - 20, 20)
        g_u, g_score = greedy_construct([0, 1, 4, 10, 18, 23, 29], t0b)
        if g_score > best_score:
            best_set = g_u
            best_score = g_score
    
    if remaining() > 20:
        t1 = min(remaining() - 5, 35)
        ext_u, ext_score = local_search(best_set, t1)
        if ext_score > best_score:
            best_set = ext_u
            best_score = ext_score
    
    if remaining() > 15:
        t2 = min(remaining() - 3, 25)
        grow_u, grow_score = grow_search(best_set, t2)
        if grow_score > best_score:
            best_set = grow_u
            best_score = grow_score
    
    if remaining() > 8:
        t3 = remaining() - 2
        ext_u, ext_score = local_search(best_set, t3)
        if ext_score > best_score:
            best_set = ext_u
            best_score = ext_score
    
    return best_set
