import math
import time
import random
import numpy as np

def construct():
    START_TIME = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    def elapsed():
        return time.time() - START_TIME
    
    def remaining():
        return DEADLINE - elapsed() - 3  # 3s safety margin
    
    def compute_score(u):
        """Compute the score for a set u. Uses FFT for efficiency on larger sets."""
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        
        n = len(u)
        # For small sets, direct computation is faster
        if n <= 20:
            arr = np.array(u, dtype=np.int64)
            sums = arr[:, None] + arr[None, :]
            diffs = arr[:, None] - arr[None, :]
            sumset_size = len(np.unique(sums))
            diffset_size = len(np.unique(diffs))
        else:
            # FFT-based approach
            # For sumset: use convolution of indicator function with itself
            # For diffset: use convolution of indicator with reversed indicator
            max_val = max_u
            # Need array of size at least 2*max_val + 1 for sums (range 0 to 2*max_val)
            # And for diffs (range -max_val to max_val)
            
            # Sumset via FFT
            size_sum = 2 * (2 * max_val + 1)  # power of 2 would be better, but let's try
            # Actually, let's use a size that's a power of 2 for FFT efficiency
            n_fft = 1
            while n_fft < 2 * (2 * max_val + 1):
                n_fft *= 2
            
            indicator = np.zeros(n_fft, dtype=np.float64)
            for val in u:
                indicator[val] = 1.0
            
            # Sumset: convolution of indicator with itself
            ft = np.fft.fft(indicator)
            sum_conv = np.fft.ifft(ft * ft).real
            sumset_size = np.sum(np.round(sum_conv) > 0.5)
            
            # Diffset: we need |a - b| for a, b in u
            # This is the same as the support of the autocorrelation
            # autocorr[k] = sum_j indicator[j] * indicator[j+k]
            # The diffset size is the number of k where autocorr[k] > 0
            diff_conv = np.fft.ifft(ft * np.conj(ft)).real
            # diff_conv[k] for k=0..n_fft-1 represents autocorrelation at lag k
            # But we need negative lags too. The autocorrelation is symmetric.
            # diff_conv[k] for k in 0..max_val gives positive lags
            # diff_conv[n_fft - k] gives negative lags (by symmetry of real autocorrelation)
            # Actually, for real signals, ifft(ft * conj(ft)) gives the circular autocorrelation
            # which is symmetric: R[k] = R[n_fft - k]
            # We need to count distinct values of a-b where a,b in u
            # This equals the number of distinct lags k (including negative) where there exist a,b in u with a-b=k
            # Which is the same as the support size of the autocorrelation function
            # But we need to be careful about circular wrap-around
            
            # Let's just count: for each pair (a,b), diff = a-b. 
            # The number of distinct diffs is what we want.
            # Using FFT: the autocorrelation R[k] = sum_x f[x] f[x+k]
            # R[k] > 0 iff there exist x, y in u with y - x = k (mod n_fft)
            # To avoid wrap-around issues, we need n_fft > 2 * max_val
            # Since n_fft >= 2*(2*max_val+1), this should be fine
            
            # Count positive and zero lags
            half = n_fft // 2
            positive_lags = np.sum(np.round(diff_conv[:half+1]) > 0.5)
            negative_lags = np.sum(np.round(diff_conv[half+1:]) > 0.5)
            diffset_size = positive_lags + negative_lags
        
        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        """Check if u is a Golomb ruler (all pairwise differences are distinct)."""
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
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
    
    # Local search: try to improve by replacing elements
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
            search_hi = current_max + max(10, n * 3)
            search_lo = 1
            
            # Try each position (except 0 at index 0)
            indices_to_try = list(range(1, n))
            random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if (time.time() - start_time) > time_limit:
                    break
                old_val = best_u[idx]
                # Try a window of values
                window = max(8, n * 3)
                lo = max(search_lo, old_val - window)
                hi = min(search_hi, old_val + window)
                
                for new_val in range(lo, hi + 1):
                    if new_val == old_val:
                        continue
                    if new_val in best_u:
                        continue
                    
                    candidate = sorted([x if x != old_val else new_val for x in best_u])
                    
                    # Quick check: must be Golomb for best ratio
                    if not is_golomb(candidate):
                        continue
                    
                    s = compute_score(candidate)
                    if s > best_score_local + 1e-9:
                        best_score_local = s
                        best_u = candidate
                        improved = True
                        break  # Restart with new best
        
        return best_u, best_score_local
    
    # Also try growing the set: add a new element
    def grow_search(base_u, time_limit):
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            current_max = max(best_u)
            n = len(best_u)
            
            # Try adding a new element
            search_hi = current_max + max(15, n * 4)
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
    
    # Allocate time to different search phases
    if remaining() > 20:
        # Phase 1: Local search on current best
        t1 = min(remaining() - 5, 40)
        ext_u, ext_score = local_search(best_set, t1)
        if ext_score > best_score:
            best_set = ext_u
            best_score = ext_score
    
    if remaining() > 15:
        # Phase 2: Try growing from best
        t2 = min(remaining() - 3, 30)
        grow_u, grow_score = grow_search(best_set, t2)
        if grow_score > best_score:
            best_set = grow_u
            best_score = grow_score
    
    if remaining() > 10:
        # Phase 3: More local search on the best found so far
        t3 = remaining() - 2
        ext_u, ext_score = local_search(best_set, t3)
        if ext_score > best_score:
            best_set = ext_u
            best_score = ext_score
    
    return best_set
