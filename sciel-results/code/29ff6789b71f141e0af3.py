import time
import numpy as np
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N_MIN = 10000
    N_MAX = 1000000

    def get_primes_up_to(limit):
        is_prime = np.ones(limit + 1, dtype=np.bool_)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i : limit+1 : i] = False
        return np.where(is_prime)[0]

    # Generate P-smooth numbers up to max_val
    def generate_smooth_numbers(primes, max_val):
        # Use a set to store unique smooth numbers
        smooth = set([1])
        # We can use a priority queue to generate in sorted order, 
        # or just expand the set. 
        # Given max_val is 10^6, the number of smooth numbers is small.
        # Let's use a simple expansion loop with a list of current numbers.
        # To be efficient, we iterate primes and multiply.
        
        # A better way: Use a heap to generate in sorted order to potentially stop early?
        # But we need all to check consecutive.
        # Let's just generate all.
        
        # Start with 1
        nums = [1]
        for p in primes:
            new_nums = []
            for n in nums:
                # Multiply by p repeatedly
                val = n
                while val * p <= max_val:
                    val *= p
                    new_nums.append(val)
            # Add new nums to the set
            smooth.update(new_nums)
            # Note: This doesn't generate all combinations correctly if we just append to nums?
            # No, we need to multiply previous smooth numbers by new prime powers.
            # Actually, the standard way is:
            # smooth = {1}
            # for p in primes:
            #   for x in list(smooth):
            #     while x*p <= max_val:
            #        x *= p
            #        smooth.add(x)
            
            # Let's re-implement correctly inside the function.
        return smooth

    # Correct smooth generator
    def gen_smooth(primes, max_val):
        smooth = set([1])
        for p in primes:
            # We must iterate over a snapshot of the current set
            # because we are adding to it
            current_smooth = list(smooth)
            for x in current_smooth:
                val = x
                while val * p <= max_val:
                    val *= p
                    smooth.add(val)
        return smooth

    # Try small primes P
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31
    # We need n >= 10000.
    
    small_primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_n = None
    best_lpf = float('inf')

    for P in small_primes_list:
        if time.time() > deadline - 10:
            break
            
        primes = [p for p in small_primes_list if p <= P]
        # Generate P-smooth numbers up to N_MAX + 1
        smooth_nums = gen_smooth(primes, N_MAX + 1)
        
        # Sort and find consecutive pairs
        sorted_smooth = sorted(smooth_nums)
        
        found_pair = False
        for i in range(len(sorted_smooth) - 1):
            if sorted_smooth[i+1] - sorted_smooth[i] == 1:
                n = sorted_smooth[i]
                if n >= N_MIN and n <= N_MAX:
                    # The LPF of n and n+1 is at most P.
                    # Actually, it is exactly the max of their LPFs, which is <= P.
                    # Since we are iterating P in increasing order, the first pair found
                    # will have the smallest possible LPF (which is P, or a divisor of P).
                    # Wait, if P=5, LPF is 5. If P=7, LPF is 7.
                    # So the first P that yields a pair gives the global minimum LPF?
                    # Yes, because any number with LPF < P would have been found in a previous step.
                    return n
        # If no pair found for this P, continue to next P

    # Fallback to sieve if no small-P smooth pair found
    limit = N_MAX + 1
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
    
    if time.time() > deadline - 5:
        return 10000
        
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
