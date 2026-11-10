import time
import math
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 90s kill

    MAX_N = 256

    # 1. Sieve Primes
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_N + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Split Primes
    # Primes > 13 cannot be part of any product p*q <= 256 with q > p.
    # Thus, they must be selected as their maximal power (which is just p itself, since p^2 > 256 for p > 16, 
    # and 17^2 = 289 > 256). Wait, 13^2 = 169 <= 256. 
    # Let's re-evaluate "large primes".
    # A prime p is "small" if it can appear in a composite number <= 256.
    # 2, 3, 5, 7, 11, 13 are small.
    # 17 is the next prime. 17 * 2 = 34. But if we use 17, we can't use 2.
    # However, the "conflict graph" approach handles all primes.
    # The ILP approach (Inspiration 1) is robust and exact.
    # The Current Program uses a mask-based branch and bound for "small primes" only,
    # assuming large primes are always taken.
    # Is it always optimal to take large primes as themselves?
    # If we take prime p > 13, we can only use p (since p^2 > 256).
    # Could we use p * q? No, because q >= 2, so p*q >= 2*17 = 34.
    # But if we use p*q, we "spend" prime q.
    # The value gained is p*q instead of p + q^k (where q^k is the max power of q used).
    # Usually q^k is much larger than q. E.g., for q=2, max power is 128.
    # p*q = 17*2 = 34. p + 128 = 17 + 128 = 145. 145 > 34.
    # So it is better to keep 2 as 128 and 17 as 17.
    # Generally, for p > 13, p*q < p + q_max_power?
    # Let's check the worst case for q. Smallest q is 2.
    # If we pair p with 2, we lose 128 (max power of 2) and gain p*2.
    # Net change: 2p - 128.
    # For p=17, 34 - 128 = -94. Bad.
    # For p=127 (largest prime < 256), 254 - 128 = 126. Good?
    # Wait, if we pair 127 and 2, we get 254.
    # If we keep them separate, we get 127 + 128 = 255.
    # 255 > 254.
    # So even for the largest prime, keeping them separate is better.
    # Therefore, ALL primes > 13 should be selected as themselves (or their max power if < 256, which is just p).
    # What about primes <= 13?
    # 13^2 = 169. 13^3 > 256. Max power is 169.
    # 11^2 = 121.
    # 7^3 = 343 > 256. 7^2 = 49.
    # 5^3 = 125.
    # 3^5 = 243.
    # 2^7 = 128.
    
    # So the "large primes" (17 and above) are fixed.
    # The "small primes" (2, 3, 5, 7, 11, 13) need to be optimized.
    # They can form composites.
    
    small_primes = [p for p in primes if p <= 13]
    large_primes = [p for p in primes if p > 13]
    
    # Large primes are always selected as themselves
    selected_large = large_primes
    sum_large = sum(selected_large)
    
    # 3. Generate Candidates for Small Primes
    # Candidates are numbers <= 256 with prime factors ONLY in small_primes.
    # We want to select a subset of these candidates such that they are pairwise coprime.
    # Since they only have small prime factors, pairwise coprime means they share no prime factors.
    # This is equivalent to: for each small prime p, at most one selected candidate is divisible by p.
    
    prime_to_idx = {p: i for i, p in enumerate(small_primes)}
    num_small_primes = len(small_primes)
    
    candidates = []
    for n in range(2, MAX_N + 1):
        temp = n
        valid = True
        mask = 0
        for p in small_primes:
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
            if temp == 1:
                break
        if temp != 1:
            continue
        # n is a candidate composed only of small primes
        # Note: n=1 is handled separately. We start from 2.
        candidates.append((n, mask))
        
    # Sort by value descending to help pruning
    candidates.sort(key=lambda x: -x[0])
    
    # 4. Solve Max Weight Independent Set using Branch and Bound
    # State: (index, used_mask, current_sum)
    # Upper bound: current_sum + sum of values of all remaining candidates that don't conflict with used_mask.
    # To make the bound tighter and faster, we can precompute suffix sums or use a greedy bound.
    # Given the small size (candidates are numbers composed of 2,3,5,7,11,13 <= 256),
    # there are not many candidates.
    
    # Let's count candidates:
    # Powers of 2: 2,4,8,16,32,64,128 (7)
    # Powers of 3: 3,9,27,81,243 (5)
    # Powers of 5: 5,25,125 (3)
    # Powers of 7: 7,49 (2)
    # Powers of 11: 11,121 (2)
    # Powers of 13: 13,169 (2)
    # Products: 2*3=6, 2*5=10, ... many more.
    # Total candidates is likely < 100.
    # Branch and bound will be very fast.
    
    best_sum = 0
    best_set = []
    
    # Precompute suffix sums for a simple bound?
    # A better bound: for each prime p not in used_mask, add the max value of any candidate divisible by p (and not conflicting with other used primes? No, just max value compatible with current mask).
    # Actually, the simplest strong bound is:
    # For each remaining candidate i >= idx, if (mask_i & used_mask) == 0, it can be added.
    # But we can't add all of them.
    # A valid upper bound is: current_sum + sum(max_val_for_prime[p] for p not in used_mask)
    # where max_val_for_prime[p] is the maximum value of any candidate divisible by p.
    # Wait, this is not quite right because a candidate might be divisible by multiple primes.
    # However, since each prime can only be used once, the sum of the max values for each unused prime is an upper bound on the additional value we can get.
    # Why? Because any set of disjoint candidates corresponds to a partition of a subset of unused primes.
    # The value of a candidate is at most the sum of the max values of its constituent primes? No.
    # Example: candidate 6 (2*3). max_val[2]=128, max_val[3]=243. Sum=371. Value of 6 is 6.
    # This bound is very loose.
    
    # Better bound:
    # Let U be the set of unused primes.
    # We want to maximize sum of selected candidates.
    # Each selected candidate uses a unique subset of U.
    # This is a packing problem.
    # A simple bound: For each unused prime p, let M_p be the maximum value of any candidate that uses p.
    # Then the max additional value is <= sum(M_p for p in U).
    # This is still loose but better than nothing.
    
    # Even better: Branch and bound with ordering.
    # Since N is small, we can just do DFS with a simple sum-of-remaining-compatible-candidates bound.
    # Bound = current_sum + sum(val for i in range(idx, N) if (mask_i & used_mask) == 0)
    # This is O(N) per node. With N~100 and depth ~10, it's fast.
    
    N_cands = len(candidates)
    
    # Precompute masks and values
    cand_vals = [c[0] for c in candidates]
    cand_masks = [c[1] for c in candidates]
    
    # Suffix sum of all values (ignoring conflicts) for a very loose initial bound?
    # No, let's just use the recursive search.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(idx, used_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        # Pruning
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)
            
        if idx == N_cands:
            return
            
        # Calculate upper bound: current_sum + sum of values of all remaining candidates that don't conflict
        # This is a valid upper bound because we can select AT MOST all of them.
        bound = current_sum
        for i in range(idx, N_cands):
            if (cand_masks[i] & used_mask) == 0:
                bound += cand_vals[i]
                
        if bound <= best_sum:
            return
            
        # Option 1: Skip current candidate
        search(idx + 1, used_mask, current_sum, current_set)
        
        # Option 2: Include current candidate if compatible
        val, mask = cand_vals[idx], cand_masks[idx]
        if (mask & used_mask) == 0:
            current_set.append(val)
            search(idx + 1, used_mask | mask, current_sum + val, current_set)
            current_set.pop()
            
    search(0, 0, 0, [])
    
    # 5. Construct Final List
    final_set = [1] + selected_large + best_set
    
    # Verify validity just in case
    # Check distinctness
    if len(final_set) != len(set(final_set)):
        # Fallback to just primes if something went wrong
        return [int(x) for x in [1] + primes]
        
    # Check range
    for x in final_set:
        if x < 1 or x > 256:
            return [int(x) for x in [1] + primes]
            
    # Check pairwise coprimality
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            if gcd(final_set[i], final_set[j]) != 1:
                # Fallback
                return [int(x) for x in [1] + primes]
                
    if time.time() > deadline:
        return [int(x) for x in [1] + primes]
        
    return [int(x) for x in final_set]
