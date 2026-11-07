import time
import numpy as np
from fractions import Fraction

def _calc_sigma_simple(n):
    """Calculate sigma(n) using trial division for verification."""
    if n == 1:
        return 1
    res = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            term = 1
            power = 1
            while temp % d == 0:
                temp //= d
                power *= d
                term += power
            res *= term
        d += 1
    if temp > 1:
        res *= (1 + temp)
    return res

def construct():
    t0 = time.time()
    LIMIT = 60.0
    SAFETY = 0.85
    hard_stop = t0 + LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    N = 10000000
    fallback = [11, 6]  # sigma(11)=12, sigma(6)=12. |11/6 - 3/2| = |1.8333 - 1.5| = 0.3333

    try:
        # 2. Compute Sigma via vectorized divisor-sum sieve
        # sigma(n) = sum of all divisors of n
        # Initialize sigma[i] = 0 for all i
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # For each prime p, add p * sigma[n//p] to sigma[n] for all multiples n of p
        # This is equivalent to: for each divisor d, add d to sigma[multiple of d]
        # Vectorized approach: for each i from 2 to N, if i is prime, update multiples
        # But we can do it more efficiently:
        # sigma[n] = sum_{d|n} d
        # We can compute this by: for each d from 1 to N, for each multiple m of d, sigma[m] += d
        # This is O(N log N) but with small constant if vectorized per d
        
        # Better: use the multiplicative property with SPF
        # Actually, let's use the direct divisor sum approach with numpy
        # For each divisor d, add d to all multiples of d
        # sigma[n] = sum_{d|n} d
        
        # Vectorized: for d in range(1, N+1): sigma[d::d] += d
        # This is O(N log N) operations but each is vectorized
        # However, this is still a Python loop over N iterations
        # Let's try a different approach: use the formula sigma(n) = product of (p^(k+1)-1)/(p-1)
        
        # Actually, the most efficient vectorized approach for sigma is:
        # Use the fact that sigma is multiplicative and compute via SPF
        # But the SPF-based DP in Python is slow.
        
        # Alternative: compute sigma using a sieve-like approach
        # sigma[n] = sum_{d|n} d
        # We can compute this by iterating d from 1 to N and adding d to all multiples
        # This is O(N log N) but with vectorized inner loop
        
        # Let's try: for d in range(1, N+1): sigma[d::d] += d
        # This should be fast enough with numpy
        
        _sigma = sigma
        for d in range(1, N + 1):
            _sigma[d::d] += d
            
        if time_left() < 15.0:
            return fallback

        # 3. Group by Sigma value
        # We need to find pairs (a, b) with sigma(a) = sigma(b) and a != b
        # that minimize |a/b - 3/2|
        
        # Get unique sigma values and their indices
        idx = np.arange(1, N + 1, dtype=np.int32)
        vals = _sigma[1:]  # sigma values for 1..N
        
        # Sort by sigma value
        order = np.argsort(vals, kind='stable')
        svals = vals[order]
        
        # Find group boundaries
        M = order.size
        change = np.empty(M, dtype=bool)
        change[0] = True
        change[1:] = svals[1:] != svals[:-1]
        starts = np.nonzero(change)[0]
        ends = np.append(starts[1:], M)
        
        best_val = float('inf')
        best_pair = None
        
        # Iterate over groups with size >= 2
        for gi in range(starts.size):
            if time_left() < 5.0:
                break
                
            s0 = int(starts[gi])
            e0 = int(ends[gi])
            length = e0 - s0
            if length < 2:
                continue
                
            # Extract the numbers in this group
            group_nums = np.sort(idx[order[s0:e0]])
            L = len(group_nums)
            
            # For each b in group, find a that minimizes |a/b - 3/2|
            # |a/b - 3/2| = |2a - 3b| / (2b)
            # We want to minimize this, so for each b, find a closest to 1.5*b
            
            # Vectorized approach:
            # For each b, target = 1.5 * b
            # Find closest a in group_nums to target
            targets = 1.5 * group_nums.astype(np.float64)
            
            # Use searchsorted to find insertion points
            j_left = np.searchsorted(group_nums, targets, side='left')
            
            # Check candidates from the right (a >= target)
            valid_left = j_left < L
            if np.any(valid_left):
                a_cands = group_nums[j_left[valid_left]]
                b_cands = group_nums[valid_left]
                
                num = np.abs(2 * a_cands - 3 * b_cands)
                mask_valid = num > 0  # exclude a/b = 3/2 exactly
                if np.any(mask_valid):
                    a_v = a_cands[mask_valid]
                    b_v = b_cands[mask_valid]
                    num_v = num[mask_valid]
                    den_v = 2 * b_v
                    val = num_v / den_v
                    
                    min_idx = np.argmin(val)
                    if val[min_idx] < best_val:
                        best_val = val[min_idx]
                        best_pair = (int(a_v[min_idx]), int(b_v[min_idx]))
                        
            # Check candidates from the left (a < target)
            valid_right = j_left > 0
            if np.any(valid_right):
                j_right = j_left[valid_right] - 1
                a_cands = group_nums[j_right]
                b_cands = group_nums[valid_right]
                
                num = np.abs(2 * a_cands - 3 * b_cands)
                mask_valid = (num > 0) & (a_cands != b_cands)
                if np.any(mask_valid):
                    a_v = a_cands[mask_valid]
                    b_v = b_cands[mask_valid]
                    num_v = num[mask_valid]
                    den_v = 2 * b_v
                    val = num_v / den_v
                    
                    min_idx = np.argmin(val)
                    if val[min_idx] < best_val:
                        best_val = val[min_idx]
                        best_pair = (int(a_v[min_idx]), int(b_v[min_idx]))

        if best_pair is None:
            return fallback
            
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= N and 1 <= b <= N):
            return fallback
        if 2 * a == 3 * b:
            return fallback
        
        # Verify the pair to ensure sigma(a) == sigma(b)
        # This protects against any bugs in the sieve calculation
        if _calc_sigma_simple(a) != _calc_sigma_simple(b):
            return fallback

        return [a, b]

    except Exception:
        return [11, 6]
