import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill
    
    N = 123457
    
    # First, compute LPF (largest prime factor) for all numbers up to N using a sieve
    # This allows us to quickly check P(n) for any n
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = p  # since we're dividing out smallest first, the last prime factor is the largest
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
    
    # Now, we want to find a, b > 0 with a + b = N that minimizes max(lpf[a], lpf[b])
    # Strategy: for each candidate bound B (increasing), check if there exists a pair
    # where both a and b have lpf <= B
    
    # Generate smooth numbers for increasing B
    def generate_smooth_numbers(prime_bound, limit):
        """Generate all numbers <= limit whose largest prime factor <= prime_bound."""
        # Find all primes <= prime_bound
        primes = []
        for x in range(2, prime_bound + 1):
            if spf[x] == x:
                primes.append(x)
        
        # Generate all products of prime powers
        result = set([1])
        for p in primes:
            new_result = set()
            for base in result:
                v = 1
                while v <= limit:
                    val = base * v
                    if val <= limit:
                        new_result.add(val)
                    v *= p
            result = new_result
        return result
    
    # Prime bounds to try, in increasing order
    prime_bounds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
    
    best_a = 1
    best_b = N - 1
    best_val = lpf[best_a] if best_a > 1 else 1
    best_val = max(best_val, lpf[best_b])
    
    for B in prime_bounds:
        if time.time() > deadline:
            break
        
        smooth_set = generate_smooth_numbers(B, N)
        
        # Check for pairs a + b = N where both are smooth
        found = False
        for a in smooth_set:
            if a > N - 1 or a < 1:
                continue
            b = N - a
            if b in smooth_set and b >= 1:
                # Found a valid pair
                found = True
                best_a = a
                best_b = b
                best_val = B
                break
        
        if found:
            # This is the minimum B (since we go in increasing order)
            break
    
    # If no smooth pair found with our bounds, fall back to brute force
    if best_val == lpf[N - 1] and best_a == 1:
        # This means we didn't find a better pair, so do a full search
        best_val = max(lpf[1], lpf[N - 1])
        best_a = 1
        for a in range(2, N):
            if time.time() > deadline:
                break
            b = N - a
            val = max(lpf[a], lpf[b])
            if val < best_val:
                best_val = val
                best_a = a
        best_b = N - best_a
    
    return [best_a, best_b]
