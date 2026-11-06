"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    # b^3 * a^2 <= limit
    # b up to limit^(1/3)
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with a moderate limit to get a quick baseline
    limit = 10**7
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        powerful = _generate_powerful(limit)
        powerful_set = set(powerful)
        
        # Count representations for all N up to limit
        # For each pair (a, b) with a <= b, a+b = N, both powerful
        # We want to find N with the most such pairs
        
        # Use numpy for efficient counting
        import numpy as np
        
        arr = np.array(powerful, dtype=np.int64)
        M = len(arr)
        
        if M > 50000:
            # Too many, reduce limit
            limit = int(limit * 0.5)
            continue
        
        # Generate all pairs a <= b where a+b <= limit
        # This is O(M^2) which might be too slow for large M
        # Use a smarter approach: for each a, find b in powerful where b >= a and b <= limit - a
        
        # Count representations using bincount on sums
        # For each a in powerful, the valid b values are powerful numbers in [a, limit - a]
        # Sum = a + b
        
        # Vectorized approach: for each a, compute sums with all b >= a
        # But this is still O(M^2) in the worst case
        
        # Alternative: use the fact that we want to count for each N
        # How many pairs (a, b) with a <= b, a+b=N, a,b in powerful
        
        # Let's use a different strategy: iterate over possible N and count
        # But that's O(limit * M) which is too slow
        
        # Best approach for moderate M: generate all pairs
        # For M up to ~10000, M^2 = 10^8 which is feasible with numpy
        
        if M <= 15000:
            # Generate upper triangular pairs
            # For each i, j >= i, sum = arr[i] + arr[j]
            # Use broadcasting in chunks to avoid memory issues
            
            counts = np.zeros(limit + 1, dtype=np.int32)
            
            chunk_size = 2000
            for i_start in range(0, M, chunk_size):
                i_end = min(i_start + chunk_size, M)
                # For rows i_start to i_end, columns j >= i
                # We need j such that arr[i] + arr[j] <= limit
                # For each i in [i_start, i_end), j ranges from i to max_j where arr[j] <= limit - arr[i]
                
                for i in range(i_start, i_end):
                    ai = arr[i]
                    if ai * 2 > limit:
                        break
                    # Find max j such that arr[j] <= limit - ai
                    # arr is sorted
                    max_j_val = limit - ai
                    # Use searchsorted
                    j_max = np.searchsorted(arr, max_j_val, side='right') - 1
                    if j_max < i:
                        continue
                    # Sums for j in [i, j_max]
                    sums = ai + arr[i:j_max+1]
                    # Count these sums
                    # Only count if sum <= limit (should be guaranteed)
                    valid_sums = sums[sums <= limit]
                    if len(valid_sums) > 0:
                        np.add.at(counts, valid_sums, 1)
            
            # Find the N with maximum count
            best_N = int(np.argmax(counts))
            best_count = int(counts[best_N])
            
            # Collect the pairs for this N
            pairs = []
            for a in powerful:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b in powerful_set and a <= b:
                    pairs.append([a, b])
            
            # Check if we should continue searching with larger limit
            if best_count >= 8:
                # Good result, try to improve with larger limit
                pass
            else:
                # Try larger limit
                pass
            
            # Try to increase limit for next iteration
            new_limit = int(limit * 2)
            if new_limit > 10**18:
                new_limit = 10**18
            
            # Estimate if next iteration would be too slow
            # Roughly, number of powerful numbers grows as ~limit^(1/2) * some factor
            # Actually powerful numbers are sparse: ~sqrt(limit) * constant
            # So M ~ O(sqrt(limit))
            # Next M ~ M * sqrt(2)
            # If M * sqrt(2) > 15000, don't increase
            if M * 1.42 > 15000:
                break
            
            limit = new_limit
            continue
        else:
            # M too large, reduce limit
            limit = int(limit * 0.5)
            if limit < 10**5:
                break
            continue
    
    # Final answer: use the best found
    # We need to recompute for the current limit if we broke out
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    
    import numpy as np
    arr = np.array(powerful, dtype=np.int64)
    M = len(arr)
    
    if M <= 15000:
        counts = np.zeros(limit + 1, dtype=np.int32)
        for i in range(M):
            ai = arr[i]
            if ai * 2 > limit:
                break
            max_j_val = limit - ai
            j_max = np.searchsorted(arr, max_j_val, side='right') - 1
            if j_max < i:
                continue
            sums = ai + arr[i:j_max+1]
            valid_sums = sums[sums <= limit]
            if len(valid_sums) > 0:
                np.add.at(counts, valid_sums, 1)
        
        best_N = int(np.argmax(counts))
        best_count = int(counts[best_N])
        
        pairs = []
        for a in powerful:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set and a <= b:
                pairs.append([a, b])
        
        return {'N': best_N, 'pairs': pairs}
    else:
        # Fallback: just return a simple result
        # Find a small N with at least one representation
        for N in range(2, min(limit, 10**6) + 1):
            pairs = []
            for a in powerful:
                if 2 * a > N:
                    break
                b = N - a
                if b in powerful_set and a <= b:
                    pairs.append([a, b])
            if pairs:
                return {'N': N, 'pairs': pairs}
        
        return {'N': 2, 'pairs': [[1, 1]]}

# EVOLVE-BLOCK-END
