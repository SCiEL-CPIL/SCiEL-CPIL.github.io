import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need smooth parts for numbers in [1001, 1000024]
    # That's 1000024 - 1001 + 1 = 1000024 numbers
    lo = 1001
    hi = 1000024
    N = hi - lo + 1
    
    # Compute smooth part for each number in [lo, hi]
    # smooth[m] = product of p^v_p(m) for p in primes
    # We'll use iterative division approach with NumPy
    
    # Create array of all numbers
    nums = np.arange(lo, hi + 1, dtype=np.int64)
    
    # smooth_part[i] will be the 23-smooth part of nums[i]
    smooth_part = np.ones(N, dtype=np.int64)
    
    for p in primes:
        # For each number, extract the p-part
        # p_part[i] = p^v_p(nums[i])
        # We can do this by repeatedly dividing by p
        # But we need to track which positions are divisible
        
        # Start with the p-part as 1 for all
        p_part = np.ones(N, dtype=np.int64)
        
        # Create a copy to work with
        remaining = nums.copy()
        
        # Iteratively extract factors of p
        power = np.int64(p)
        while True:
            # Check which remaining numbers are divisible by power
            mask = (remaining % power == 0)
            if not np.any(mask):
                break
            # For those divisible, the p-part gets multiplied by power
            # But we need to be careful: we want the full p^v, not just one factor
            # Better approach: for each number, count how many times p divides it
            # Actually, let's use a different approach:
            # p_part[i] = p^(v_p(nums[i]))
            # We can compute this by dividing nums[i] by p repeatedly until not divisible
            # But that's slow in pure Python. Let's use the sieve-like approach.
            break
        
        # Better: use the sieve approach
        # For each power of p, mark which numbers are divisible by that power
        # but not by the next power
        p_power = np.int64(p)
        while p_power <= hi:
            # Numbers divisible by p_power
            mask = (nums % p_power == 0)
            if np.any(mask):
                # Among those, which are NOT divisible by p_power * p?
                if p_power * p <= hi:
                    mask_not_next = (nums % (p_power * p) != 0)
                    # For numbers divisible by p_power but not p_power*p,
                    # their p-part is exactly p_power
                    target = mask & mask_not_next
                    if np.any(target):
                        smooth_part[target] *= p_power
                else:
                    # All numbers divisible by p_power have p-part = p_power
                    smooth_part[mask] *= p_power
            p_power *= p
    
    # Now smooth_part[i] is the 23-smooth part of nums[i] = lo + i
    
    # We need to find n in [1000, 1000000] such that the window [n+1, n+24]
    # has minimum distinct smooth parts.
    # Window for n is indices [n+1-lo, n+24-lo] = [n+1-1001, n+24-1001] = [n-1000, n+23-1000]
    # So window starts at index n-1000 and ends at index n+23-1000
    # n ranges from 1000 to 1000000
    # Window start index: 0 to 999000
    # Window end index: 23 to 999023
    
    # Let's find the best window by checking for runs of equal smooth parts
    # First, check if there's any window where all 24 values are the same
    
    # For efficiency, let's first check if all 24 are equal
    # This is equivalent to: smooth_part[i] == smooth_part[i+1] == ... == smooth_part[i+23]
    
    # We can compute this by checking consecutive equalities
    # equal[i] = 1 if smooth_part[i] == smooth_part[i+1], else 0
    # Then a window starting at i has all equal iff equal[i], equal[i+1], ..., equal[i+22] are all 1
    
    # Compute consecutive equality
    equal = (smooth_part[:-1] == smooth_part[1:]).astype(np.int8)
    
    # For a window of length 24 starting at index i, we need equal[i..i+22] all 1
    # Use cumulative sum to count consecutive 1s
    # cumsum[i] = number of 1s in equal[0..i-1]
    cumsum = np.zeros(len(equal) + 1, dtype=np.int32)
    np.cumsum(equal, out=cumsum[1:])
    
    # For window starting at i (0-indexed in smooth_part), we need equal[i..i+22] all 1
    # That means cumsum[i+23] - cumsum[i] == 23
    # Window start index i corresponds to n = 1000 + i
    # n must be in [1000, 1000000], so i in [0, 999000]
    
    max_start = 1000000 - 1000  # = 999000
    valid_starts = np.arange(0, min(max_start + 1, len(equal) - 22 + 1))
    
    if len(valid_starts) > 0:
        # Check which windows have all 23 consecutive equalities
        # cumsum[i+23] - cumsum[i] == 23
        idx = valid_starts
        counts = cumsum[idx + 23] - cumsum[idx]
        
        # Find windows where all are equal
        all_equal_mask = (counts == 23)
        
        if np.any(all_equal_mask):
            # Found a window with all 24 smooth parts equal!
            best_idx = np.where(all_equal_mask)[0][0]
            return 1000 + best_idx
    
    # If no window with all equal, let's try to find the minimum distinct count
    # This is harder to vectorize efficiently. Let's use a sliding window approach
    # but with NumPy for the smooth part lookup.
    
    # For now, let's do a simple scan with early termination
    # We'll check windows in chunks
    
    best_n = 1000
    best_score = 25
    
    # Use a more efficient approach: for each starting position, count distinct values
    # in the window of 24. We can do this in chunks.
    
    chunk_size = 50000
    for chunk_start in range(0, min(max_start + 1, len(smooth_part) - 24), chunk_size):
        if time.time() > deadline:
            break
        
        chunk_end = min(chunk_start + chunk_size, max_start + 1, len(smooth_part) - 24)
        
        # For each starting position in this chunk, get the 24-element window
        # and count distinct values
        for i in range(chunk_start, chunk_end):
            window = smooth_part[i:i+24]
            distinct = len(np.unique(window))
            if distinct < best_score:
                best_score = distinct
                best_n = 1000 + i
                if best_score == 1:
                    return best_n
            elif distinct == best_score:
                # Tie, keep earlier n (or we could keep any)
                pass
    
    return best_n
