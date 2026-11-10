import time
import math

def construct():
    N = 123457
    
    # Precompute smallest prime factor (SPF) for all numbers up to N
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_p = 1
        while n > 1:
            p = spf[n]
            max_p = p
            while n % p == 0:
                n //= p
        return max_p

    # Generate primes up to N
    primes = [i for i in range(2, N + 1) if spf[i] == i]
    
    # Strategy: Try increasing sets of primes.
    # A pair (a, b) is valid for a prime set S if all prime factors of a and b are in S.
    # We want the smallest max prime factor.
    # This is equivalent to finding the smallest k such that there exists a pair (a,b)
    # where all prime factors are <= p_k.
    
    # We can generate all "smooth" numbers for a given set of primes.
    # Then check if a and N-a are both smooth.
    
    best_val = float('inf')
    best_pair = [1, N - 1]
    
    start_time = time.time()
    deadline = start_time + 80
    
    # Iterate over possible max prime factors (primes)
    # We start from small primes and go up.
    # The first time we find a valid pair, that max prime is the minimum possible.
    
    for i, p_max in enumerate(primes):
        if time.time() > deadline:
            break
            
        # Generate all numbers <= N with prime factors only from primes[0..i]
        # Use a set for O(1) lookup
        smooth_numbers = set()
        
        # DFS to generate smooth numbers
        def generate(idx, current_val):
            if current_val > N:
                return
            if current_val > 0:
                smooth_numbers.add(current_val)
            if idx >= len(primes[0:i+1]):
                return
            p = primes[0:i+1][idx]
            # Multiply by p as many times as possible
            val = current_val
            while val <= N:
                generate(idx + 1, val)
                if val > N // p:
                    break
                val *= p
        
        # Start with 1
        generate(0, 1)
        
        # Check for pair
        found = False
        for a in smooth_numbers:
            b = N - a
            if b in smooth_numbers:
                # Found a valid pair
                # The max prime factor is p_max
                # But we should verify P(a) and P(b) just in case?
                # Actually, by construction, all factors are <= p_max.
                # So max(P(a), P(b)) <= p_max.
                # Is it exactly p_max? Maybe not, but it's <= p_max.
                # Since we iterate p_max in increasing order, the first found is minimal.
                best_val = p_max
                best_pair = [a, b]
                found = True
                break
        
        if found:
            break

    # Fallback: if no pair found with small primes, do full scan
    # (This should not happen if the list of primes is complete)
    if best_val == float('inf'):
        for a in range(1, N // 2 + 1):
            if time.time() > deadline:
                break
            b = N - a
            pa = largest_prime_factor(a)
            pb = largest_prime_factor(b)
            val = max(pa, pb)
            if val < best_val:
                best_val = val
                best_pair = [a, b]
                
    return best_pair
