"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of interest
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW_SIZE = 24
    
    # We need to evaluate n in [N_MIN, N_MAX].
    # The numbers involved are n+1 to n+24.
    # Max number is N_MAX + 24.
    MAX_VAL = N_MAX + WINDOW_SIZE
    
    # Precompute the "small part" for all numbers up to MAX_VAL.
    # small_part[x] is the largest divisor of x composed of primes in 'primes'.
    # We can compute this by initializing an array of 1s and multiplying by p 
    # for every multiple of p, extracting the full power of p.
    
    # To be efficient, we can use a sieve-like approach.
    # Initialize array of 1s.
    small_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # For each prime p, we want to multiply small_part[m] by p^k 
    # where m is a multiple of p, and k is the exponent of p in m.
    # It is faster to iterate over multiples.
    
    for p in primes:
        # Create a mask of indices that are multiples of p
        # indices = p, 2p, 3p, ...
        indices = np.arange(p, MAX_VAL + 1, p)
        
        # For each multiple m, we need to divide m by p until it's no longer divisible.
        # Let's process this in a vectorized way if possible, or loop if not.
        # Looping over ~1e6/p elements for each prime is feasible.
        # Total iterations approx N * sum(1/p) ~ 1e6 * 1.5 = 1.5e6.
        
        # However, we need to update small_part[m] by multiplying it by the p-part of m.
        # Let's extract the p-part of each multiple.
        
        # To avoid Python loops inside the prime loop which might be slow,
        # we can do:
        # val = indices
        # while val % p == 0:
        #   small_part[indices] *= p
        #   val = val // p
        # But val changes per element.
        
        # Alternative: 
        # For a specific prime p, let m = k * p^e.
        # The p-part is p^e.
        # We can compute p^e for each multiple.
        
        # Let's just do a simple loop for each prime, updating the array.
        # Since N is 1e6, and there are 9 primes, this is small enough.
        
        # Optimization: 
        # Instead of checking divisibility repeatedly, we can just divide the index by p 
        # and multiply the small_part by p.
        
        # We need to be careful with vectorized operations on changing values.
        # Let's use a temporary array for the current multiples.
        
        # Actually, a pure Python loop over 1.5e6 elements might take ~0.1-0.5s. 
        # 9 primes * 0.5s = 4.5s. This is acceptable.
        
        # Let's try a vectorized approach for the extraction.
        # For each prime p:
        #   multiples = np.arange(p, MAX_VAL+1, p)
        #   # We want to multiply small_part[multiples] by the p-adic valuation of multiples.
        #   # Let's compute the p-part for all these multiples.
        
        # Efficient way:
        #   curr = multiples.copy()
        #   while True:
        #       mask = (curr % p == 0)
        #       if not np.any(mask): break
        #       small_part[multiples[mask]] *= p
        #       curr[mask] //= p
        
        # This loop runs log_p(MAX_VAL) times. For p=2, ~20 times.
        # 20 * 1e6 operations (vectorized) is very fast.
        
        curr = np.arange(p, MAX_VAL + 1, p)
        # We need to update small_part at these indices.
        # But small_part is a flat array.
        # Let's keep track of the original indices.
        orig_indices = curr.copy()
        
        while True:
            mask = (curr % p == 0)
            if not np.any(mask):
                break
            # Multiply the small_part at the masked original indices by p
            small_part[orig_indices[mask]] *= p
            # Divide the current values by p to check for further powers
            curr[mask] //= p
            
    # Now small_part[i] contains the product of all prime factors < 24 of i.
    # We need to find n in [1000, 1000000] such that the set of distinct primes 
    # dividing {small_part[n+1], ..., small_part[n+24]} is minimized.
    
    # Note: If small_part[x] is 1, it contributes no primes.
    # We need to count distinct primes in the union of prime factors of these 24 small_parts.
    
    # Since the primes are fixed and small (9 primes), we can represent the set of primes
    # for a number as a bitmask (9 bits).
    # prime_mask[x] will be a 9-bit integer where bit i is 1 if primes[i] divides x.
    
    prime_mask = np.zeros(MAX_VAL + 1, dtype=np.uint8)
    
    for i, p in enumerate(primes):
        # If p divides x, then p divides small_part[x].
        # So we just need to check if x is divisible by p?
        # Yes, because small_part[x] is the part of x made of these primes.
        # If p|x, then p is in small_part[x].
        # So prime_mask[x] has bit i set if x % p == 0.
        
        # This is much simpler! We don't even need to calculate the full small_part value 
        # to know the *distinct primes*. We just need to know *which* primes divide x.
        # The task asks to "extract its complete factor... MINIMIZE the number of distinct extracted factors".
        # The "extracted factor" for a number is the part composed of small primes.
        # The "distinct extracted factors" likely refers to the distinct prime values involved in these factors across the window.
        # Re-reading: "MINIMIZE the number of distinct extracted factors".
        # Does "factor" mean the composite number (e.g., 12) or the prime (e.g., 2, 3)?
        # "extract its complete factor composed of primes..." -> This is a number, e.g., 12.
        # "MINIMIZE the number of distinct extracted factors" -> Distinct numbers? Or distinct primes?
        # Usually "factors" in this context implies the prime components or the values themselves.
        # Let's look at the phrasing: "number of distinct extracted factors".
        # If n+1=12 (factor 12), n+2=18 (factor 18). Distinct factors: {12, 18}. Count 2.
        # If it meant primes: {2,3}. Count 2.
        # If n+1=4 (factor 4), n+2=8 (factor 8). Distinct factors: {4,8}. Count 2. Primes: {2}. Count 1.
        # The source mentions "uniform lower bound on this minimum".
        # In number theory contexts involving "small prime parts", minimizing distinct primes is a common hard problem.
        # Minimizing distinct composite values is also possible but less standard.
        # However, "distinct extracted factors" usually refers to the values themselves if not specified as "prime factors".
        # BUT, looking at similar problems (e.g., "smooth numbers"), the "smooth part" is the value.
        # Let's assume it means distinct **prime** factors for now, as "factor" can be synonymous with prime factor in loose speech, 
        # OR it means the distinct values of the smooth parts.
        # Let's check the score of the seed. Seed returns 1000.
        # Let's calculate both for a few small windows to see which is more "interesting" or likely to have a low minimum.
        # Actually, the bitmask approach works for distinct primes.
        # If it's distinct values, we need to store the values.
        
        # Let's assume "distinct extracted factors" means distinct **primes**. Why?
        # "composed of primes... minimize the number of distinct... factors".
        # If I extract the factor 12, the factors of 12 are 2,2,3. Distinct factors of the extracted part?
        # Or distinct extracted parts?
        # "extract its complete factor... minimize the number of distinct extracted factors".
        # Grammatically: "extracted factors" are the objects extracted. The object extracted is "its complete factor composed of primes...".
        # So it likely means distinct **values** of the smooth parts.
        
        # Let's re-evaluate. If it's values, we need to count distinct integers in the set {sp[n+1]...sp[n+24]}.
        # If it's primes, we count distinct primes in the union of prime sets.
        
        # Let's look at the "source minimum f(n,t)".
        # If t=24, and we look for n such that n+1..n+24 have few distinct small-prime factors.
        # If it's primes, the max distinct primes is 9.
        # If it's values, the values can be large (up to 24^9? No, the smooth part is <= the number).
        # Minimizing distinct values is harder to "minimize" to a small number because values are unique often.
        # Minimizing distinct primes allows for a "prime-poor" window.
        
        # Let's assume **distinct primes**. This is a standard metric in such combinatorial number theory tasks (e.g., finding intervals with few prime factors).
        
        # We can compute the bitmask for each number.
        pass # We will compute bitmasks below.

    # Recalculating bitmasks efficiently
    prime_bitmask = np.zeros(MAX_VAL + 1, dtype=np.uint8)
    for i, p in enumerate(primes):
        # Bit i is set if x is divisible by p
        # We can use boolean indexing
        prime_bitmask[p::p] |= (1 << i)
        
    # Now for each window n, we want the OR of prime_bitmask[n+1 ... n+24].
    # Let's use a sliding window sum of bits? No, OR is not additive.
    # But we can maintain a count of how many numbers in the window have bit i set.
    # If count[i] > 0, then bit i is in the union.
    
    # Initialize counts for the first window [1001, 1024] (indices 1001 to 1024)
    # Window starts at n. Numbers are n+1 to n+24.
    # Indices in array: n+1 to n+24.
    
    start_idx = N_MIN + 1
    end_idx = N_MIN + WINDOW_SIZE
    
    # Get the bitmasks for the first window
    window_masks = prime_bitmask[start_idx:end_idx+1]
    
    # Count bits
    # We have 9 bits.
    # counts[i] = number of masks in window_masks with bit i set.
    counts = np.zeros(9, dtype=np.int32)
    
    for i in range(9):
        counts[i] = np.sum((window_masks >> i) & 1)
        
    # Number of distinct primes is the number of non-zero counts.
    current_distinct = np.sum(counts > 0)
    best_distinct = current_distinct
    best_n = N_MIN
    
    # Slide the window
    # Next window starts at n+1.
    # Remove index n+1, add index n+25.
    
    # We need to do this for n from N_MIN to N_MAX.
    # Total steps: 1e6.
    # Each step: update 9 counts.
    # 1e6 * 9 = 9e6 operations. In Python, this loop might be slow (~1s-5s).
    # Let's try to vectorize the sliding part or use a faster loop.
    
    # Alternative: 
    # We can precompute the "contribution" of each number.
    # But the OR operation makes it tricky.
    # The counts method is robust.
    
    # Let's optimize the loop.
    # We can update counts by subtracting the leaving mask and adding the entering mask.
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
            
        # Current n is being evaluated.
        # We already evaluated N_MIN in the init block?
        # Let's restructure to handle all n in the loop.
        
        # Actually, the init block set up n=N_MIN.
        # Let's just loop n from N_MIN to N_MAX.
        
        # Wait, the init block calculated for N_MIN.
        # Let's just loop n from N_MIN+1 to N_MAX, updating from previous.
        
        if n == N_MIN:
            continue # Already done in init? No, let's just do it inside loop for consistency or restart.
            # Restarting is safer.
            
    # Let's restart the loop logic cleanly.
    
    # Reset counts for n=N_MIN
    start_idx = N_MIN + 1
    end_idx = N_MIN + WINDOW_SIZE
    window_masks = prime_bitmask[start_idx:end_idx+1]
    counts = np.zeros(9, dtype=np.int32)
    for i in range(9):
        counts[i] = np.sum((window_masks >> i) & 1)
    current_distinct = np.sum(counts > 0)
    best_distinct = current_distinct
    best_n = N_MIN
    
    # Slide
    # Leaving index: n+1
    # Entering index: n+25
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() > deadline:
            break
            
        # Remove prime_bitmask[n] (which was n+1 in previous window? No.)
        # Previous window for n-1: indices (n-1)+1 to (n-1)+24 => n to n+23.
        # Current window for n: indices n+1 to n+24.
        # Leaving: index n.
        # Entering: index n+24.
        
        leaving_mask = prime_bitmask[n]
        entering_mask = prime_bitmask[n + WINDOW_SIZE]
        
        # Update counts
        # For each bit, if leaving has it, decrement. If entering has it, increment.
        # We can do this vectorized on the 9-bit array?
        # Or just loop 9 times.
        
        # Vectorized update:
        # bits_leaving = (leaving_mask >> np.arange(9)) & 1
        # bits_entering = (entering_mask >> np.arange(9)) & 1
        # counts += bits_entering - bits_leaving
        
        # This creates small arrays each time. Might be slower than loop?
        # Let's try the vectorized way, it's usually faster in NumPy for small arrays if overhead is low.
        
        bits_idx = np.arange(9)
        bits_leaving = (leaving_mask >> bits_idx) & 1
        bits_entering = (entering_mask >> bits_idx) & 1
        
        counts += bits_entering - bits_leaving
        
        current_distinct = np.sum(counts > 0)
        
        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
