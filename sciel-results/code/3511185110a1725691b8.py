import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 10000000
    
    # Compute sigma(n) for all n in [1, N] using NumPy
    # sigma = [0] * (N + 1) is too slow in pure python
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # for i in range(1, N + 1):
    #     sigma[i::i] += i
    # This loop in Python is slow. We need to do it in chunks or use a different method?
    # Actually, the loop `for i in range(1, N+1): sigma[i::i] += i` involves N iterations.
    # Each iteration does a vectorized addition.
    # The overhead of the Python loop is N = 10^7 iterations.
    # 10^7 iterations in Python might take ~1-5 seconds just for loop overhead.
    # The vectorized operations sum to N log N ~ 1.6 * 10^8 elements.
    # NumPy can do ~10^8-10^9 ops/sec. So this should be feasible.
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [11, 6]

    best_a = 11
    best_b = 6
    best_err_num = 2 # |2*11 - 3*6| = |22-18| = 4? No. |11/6 - 3/2| = 1/3. |2*11 - 3*6| = 4. 4 / (2*6) = 4/12 = 1/3.
    best_err_den = 12 # 2*6

    # Iterate b from N down to 1
    # We want to minimize |2a - 3b| / (2b)
    # Let k = 2a - 3b. We want small |k| and large b.
    # For a fixed b, a must be integer.
    # a = (3b + k) / 2.
    # 3b + k must be even.
    # We check a few candidates for a near 1.5b.
    
    # To speed up, we can break early if we find an error of 1/(2*max_b_found) ?
    # No, we don't know max_b_found yet. But if we find |k|=1, that's the smallest possible non-zero numerator.
    # If we find |k|=1 with b close to N, it's likely optimal.
    # The absolute minimum error is 1 / (2 * max_possible_b).
    # If we find a pair with |2a-3b| = 1, we can't do better than 1/(2b).
    # So if we find |k|=1, we should continue searching for larger b?
    # No, we are iterating b downwards. So the first |k|=1 we find has the largest b among those checked so far.
    # But there might be a larger b later? No, we go N down to 1. So the first one found has the largest b.
    # So if we find |k|=1, we can stop?
    # Wait. If we find |k|=1 at b=9,999,999, the error is 1/(2*9,999,999).
    # Could there be a pair with b=10,000,000 and |k|=1?
    # We iterate b from N down to 1. So we check N first.
    # If we check b=N and find no match, then N-1, etc.
    # So the first match with |k|=1 will have the largest b for which |k|=1 is possible.
    # Is it possible that a smaller b has a smaller error?
    # Error = |k| / (2b).
    # If we have |k|=1 at b1 and |k|=1 at b2 with b1 > b2, then error1 < error2.
    # So yes, finding the largest b with |k|=1 is optimal among |k|=1 solutions.
    # But what if |k|=2 at a much larger b?
    # Error = 2 / (2b) = 1/b.
    # Compare 1/b1 (for k=1) and 1/b2 (for k=2).
    # If b2 > 2*b1, then 1/b2 < 1/b1.
    # So we cannot stop at the first |k|=1. We must check all b?
    # However, the probability of finding a match decreases.
    # Let's just run the loop and keep the best.
    # The loop is N iterations. Inner loop is constant.
    # N = 10^7. This might be too slow in Python (10^7 iterations ~ 1-2 seconds? Maybe more with numpy array access).
    # Accessing numpy array elements in a Python loop is slow.
    # sigma[b] is a numpy scalar. Comparing sigma[a] == sigma[b] involves numpy overhead.
    
    # Better approach:
    # Group numbers by sigma value?
    # Dict: sigma_val -> list of numbers.
    # But sigma values can be large.
    # And we need to check pairs (a,b) from different lists (same sigma val).
    # For each sigma value S, let L be the list of numbers n such that sigma(n)=S.
    # We want a, b in L such that a/b approx 1.5.
    # Sort L. For each b in L, binary search for a in L near 1.5b.
    # This is much faster if the lists are small.
    # How large are the lists?
    # Average size of a sigma-fiber?
    # Most sigma values are unique or have very few preimages.
    # Only highly composite-like numbers have many preimages?
    # Actually, sigma(n) is multiplicative.
    # Let's try the dict approach.
    
    # 1. Build dict: val -> list of n
    # This takes O(N) memory and time.
    # 2. For each list, check pairs.
    
    # Let's implement this.
    
    # Clear sigma array to save memory? No, we need it to build dict.
    # Actually, we can build the dict directly from the sigma array.
    
    from collections import defaultdict
    
    sigma_map = defaultdict(list)
    for n in range(1, N + 1):
        sigma_map[sigma[n]].append(n)
        
    if time.time() > deadline:
        return [11, 6]

    # Now iterate over the map
    # We want to minimize |2a - 3b| / (2b)
    # Note: a and b are in the same list (same sigma).
    # We can swap a and b? No, a/b is directional.
    # But sigma(a)=sigma(b) is symmetric.
    # We want a approx 1.5 b.
    # So a > b usually.
    
    for val, nums in sigma_map.items():
        if time.time() > deadline - 5:
            break
            
        if len(nums) < 2:
            continue
            
        # Sort nums? They are added in increasing order of n, so they are already sorted.
        # For each b in nums, find a in nums such that a/b approx 1.5.
        # a approx 1.5 b.
        # We can use binary search.
        
        # To optimize, we can just check neighbors in the sorted list?
        # If a/b is close to 1.5, a and b are close in magnitude.
        # In the sorted list, a and b will be close in index if the density is uniform.
        # But sigma distribution is not uniform.
        
        # Let's use binary search.
        import bisect
        
        for i, b in enumerate(nums):
            # Target a = 1.5 * b
            target = 1.5 * b
            # Find position in nums
            pos = bisect.bisect_left(nums, target)
            
            # Check pos and pos-1
            for idx in [pos, pos-1]:
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    if a == b:
                        continue
                    # Check if a/b is close to 1.5
                    # Error = |2a - 3b| / (2b)
                    err_num = abs(2*a - 3*b)
                    if err_num == 0:
                        continue # Invalid
                    err_den = 2*b
                    
                    # Compare with best
                    # err_num / err_den < best_err_num / best_err_den
                    # err_num * best_err_den < best_err_num * err_den
                    if best_err_num is None or (err_num * best_err_den < best_err_num * err_den):
                        best_err_num = err_num
                        best_err_den = err_den
                        best_a = a
                        best_b = b
                        
    if best_a is not None:
        return [int(best_a), int(best_b)]
        
    return [11, 6]
