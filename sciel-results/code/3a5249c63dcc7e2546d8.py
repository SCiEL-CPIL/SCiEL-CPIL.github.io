import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need n in [1000, 1000000], and we look at n+1 to n+24
    # So we need smooth parts for numbers in [1001, 1000024]
    # Let's compute smooth parts for a large chunk at a time
    
    def compute_smooth_parts(arr):
        """Compute 23-smooth part for each element in arr using NumPy."""
        result = arr.astype(np.int64).copy()
        for p in primes:
            # Repeatedly divide by p
            while True:
                mask = (result % p == 0)
                if not mask.any():
                    break
                result[mask] //= p
        return result
    
    best_score = 999
    best_n = 1000
    
    # Process in segments to stay within memory and time
    # Each segment covers a range of n values
    # For n in [lo, hi], we need smooth parts of [lo+1, hi+24]
    
    segment_size = 50000  # n values per segment
    
    lo = 1000
    hi_max = 1000000
    
    while lo <= hi_max:
        if time.time() > deadline:
            break
        
        hi = min(lo + segment_size - 1, hi_max)
        
        # We need smooth parts for numbers lo+1 to hi+24
        nums_lo = lo + 1
        nums_hi = hi + 24
        
        # Create array of numbers
        arr = np.arange(nums_lo, nums_hi + 1, dtype=np.int64)
        
        # Compute smooth parts
        smooth = compute_smooth_parts(arr)
        
        # Now for each n in [lo, hi], the 24 smooth parts are smooth[n-lo] to smooth[n-lo+23]
        # We need to count distinct values in each window of 24
        
        # Use a sliding window approach with a dictionary for counts
        # But for speed, let's use a different approach:
        # For each position, we want the number of distinct values in smooth[i:i+24]
        
        # Efficient approach: use the fact that we can track counts
        # But with NumPy, we can do this with a trick:
        # For each unique value, find positions where it appears, then for each window,
        # count how many unique values are present.
        
        # Alternative: since the window is small (24), and we have many windows,
        # let's use a sliding window with a count dictionary, but in C-speed via NumPy tricks.
        
        # Actually, let's just use a Python loop over windows but with optimized inner logic.
        # The number of windows per segment is ~50000, which is manageable.
        
        n_windows = hi - lo + 1
        # smooth has length (hi+24) - (lo+1) + 1 = hi - lo + 24
        # Window for n=lo: smooth[0:24]
        # Window for n=lo+1: smooth[1:25]
        # ...
        # Window for n=hi: smooth[hi-lo : hi-lo+24]
        
        # Use a sliding window count approach
        from collections import defaultdict
        counts = defaultdict(int)
        
        # Initialize first window
        for j in range(24):
            counts[smooth[j]] += 1
        
        distinct_count = len(counts)
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = lo
        
        # Slide window
        for i in range(1, n_windows):
            # Remove element leaving the window
            old_val = smooth[i - 1]
            counts[old_val] -= 1
            if counts[old_val] == 0:
                del counts[old_val]
            
            # Add element entering the window
            new_val = smooth[i + 23]
            counts[new_val] += 1
            
            distinct_count = len(counts)
            if distinct_count < best_score:
                best_score = distinct_count
                best_n = lo + i
            
            if best_score == 1:
                return best_n  # Can't do better than 1
        
        lo = hi + 1
    
    return best_n
