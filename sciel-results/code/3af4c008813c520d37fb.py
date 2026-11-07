import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    MAX_VAL = 10210
    MIN_VAL = 211
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Get all primes up to MAX_VAL
    primes = [i for i in range(2, MAX_VAL + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)
    
    # Compute bitmask representation for each number
    # Bit i is set if prime i appears with odd exponent in n
    def compute_bitmask(n):
        mask = 0
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        return mask
    
    # Compute bitmask for 210
    target_mask = compute_bitmask(210)
    
    # Precompute bitmasks for all numbers
    bitmask_cache = {}
    for i in range(MIN_VAL, MAX_VAL + 1):
        bitmask_cache[i] = compute_bitmask(i)
    
    # Build reverse map: bitmask -> list of numbers with that bitmask
    mask_to_nums = defaultdict(list)
    for i in range(MIN_VAL, MAX_VAL + 1):
        mask_to_nums[bitmask_cache[i]].append(i)
    for v in mask_to_nums:
        mask_to_nums[v].sort()
    
    # Search for minimal maximum element
    # Try M from MIN_VAL upward, check if there's a subset of {MIN_VAL, ..., M-1}
    # whose XOR with bitmask_cache[M] equals target_mask
    
    best_sol = None
    best_raw = MAX_VAL - 210  # worst case
    
    # First, check for single element solutions
    for i in range(MIN_VAL, MAX_VAL + 1):
        if bitmask_cache[i] == target_mask:
            if i - 210 < best_raw:
                best_raw = i - 210
                best_sol = [i]
    
    # Check for pairs
    for i in range(MIN_VAL, MAX_VAL + 1):
        if i - 210 >= best_raw:
            break
        if time.time() > deadline:
            break
        needed_mask = bitmask_cache[i] ^ target_mask
        if needed_mask in mask_to_nums:
            nums = mask_to_nums[needed_mask]
            # Find smallest j > i
            for j in nums:
                if j > i and j - 210 < best_raw:
                    best_raw = j - 210
                    best_sol = [i, j]
                    break
    
    # Check for triplets using F2 linear algebra approach
    # For each M, we want to find a, c < M such that
    # bitmask_cache[a] ^ bitmask_cache[c] ^ bitmask_cache[M] == target_mask
    # i.e., bitmask_cache[a] ^ bitmask_cache[c] == target_mask ^ bitmask_cache[M]
    
    for M in range(MIN_VAL, MAX_VAL + 1):
        if M - 210 >= best_raw:
            break
        if time.time() > deadline:
            break
        
        target_for_pair = target_mask ^ bitmask_cache[M]
        
        # We need to find a, c in [MIN_VAL, M-1] such that
        # bitmask_cache[a] ^ bitmask_cache[c] == target_for_pair
        # This is equivalent to finding a such that
        # bitmask_cache[c] == target_for_pair ^ bitmask_cache[a]
        
        # Use a dictionary to track seen masks
        seen = {}  # mask -> smallest number with that mask
        found = False
        
        for a in range(MIN_VAL, M):
            if time.time() > deadline:
                break
            mask_a = bitmask_cache[a]
            needed_c_mask = target_for_pair ^ mask_a
            
            if needed_c_mask in seen:
                c = seen[needed_c_mask]
                if c != a and c < M:
                    # Verify the solution
                    prod = a * c * M * 210
                    k = math.isqrt(prod)
                    if k * k == prod:
                        best_raw = M - 210
                        best_sol = [a, c, M]
                        found = True
                        break
            
            # Store the smallest number for this mask
            if needed_c_mask not in seen or a < seen[needed_c_mask]:
                # Actually, we want to store mask_a -> a, not needed_c_mask
                pass
        
        # Better approach: build map as we go
        if not found:
            mask_to_smallest = {}
            for a in range(MIN_VAL, M):
                if time.time() > deadline:
                    break
                mask_a = bitmask_cache[a]
                needed_c_mask = target_for_pair ^ mask_a
                
                if needed_c_mask in mask_to_smallest:
                    c = mask_to_smallest[needed_c_mask]
                    if c != a:
                        prod = a * c * M * 210
                        k = math.isqrt(prod)
                        if k * k == prod:
                            best_raw = M - 210
                            best_sol = [a, c, M]
                            found = True
                            break
                
                if mask_a not in mask_to_smallest or a < mask_to_smallest[mask_a]:
                    mask_to_smallest[mask_a] = a
        
        if found:
            break
    
    # If no solution found with the above, use a fallback
    if best_sol is None:
        best_sol = [840]
        best_raw = 630
    
    result = [int(x) for x in best_sol]
    return result
