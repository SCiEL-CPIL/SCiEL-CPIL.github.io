import time
import math

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # We need primes to iterate P.
    # The largest prime factor of a number <= 1000000 is <= 1000000.
    # But we expect P to be small.
    # Let's get primes up to 10000 just in case, but we'll stop early.
    primes = _sieve_primes(10000)
    
    lower_bound = 10000
    upper_bound = 1000000
    
    # Iterate over possible P (primes)
    # P must be at least 2.
    for P in primes:
        if time.time() > deadline:
            break
            
        # Generate all P-smooth numbers in [lower_bound, upper_bound]
        # Primes to use: all primes <= P
        # Find index of P in primes list
        # Since primes is sorted, we can just take prefix
        # But we need to know which primes are <= P
        # Actually, we can just iterate primes list and stop when prime > P
        active_primes = []
        for p in primes:
            if p > P:
                break
            active_primes.append(p)
            
        # Precompute powers for each active prime
        # powers[i] is list of powers of active_primes[i] <= upper_bound
        powers = []
        for p in active_primes:
            pw = [1]
            val = 1
            while val * p <= upper_bound:
                val *= p
                pw.append(val)
            powers.append(pw)
        
        # Generate products
        smooth_numbers = set()
        
        # Recursive generation
        # To avoid deep recursion or function call overhead, we can use iterative backtracking
        # Or simple recursion since depth is small (number of active primes)
        
        def gen(idx, current_val):
            if current_val > upper_bound:
                return
            if idx == len(active_primes):
                if lower_bound <= current_val <= upper_bound:
                    smooth_numbers.add(current_val)
                return
            
            p_powers = powers[idx]
            for pv in p_powers:
                next_val = current_val * pv
                if next_val > upper_bound:
                    break
                gen(idx + 1, next_val)
                
        gen(0, 1)
        
        # Check for consecutive pairs
        # Sort the numbers to check consecutiveness efficiently?
        # Or just iterate through the set and check if n+1 is in set.
        # Set lookup is O(1).
        for n in smooth_numbers:
            if n + 1 in smooth_numbers:
                return n
        
    # Fallback if no pair found (should not happen)
    # We can try a simple search for the best n
    # But this is unlikely to be reached if P is small.
    # For safety, return a known smooth-ish number or just 10000
    return 10000

# EVOLVE-BLOCK-END
