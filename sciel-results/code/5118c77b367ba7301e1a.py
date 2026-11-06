"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
    
    def generate_smooth(prime_list, limit):
        """Generate all numbers <= limit with all prime factors in prime_list,
        returning dict {number: largest_prime_factor}"""
        smooth = {}
        def gen(idx, current, max_pf):
            if time.time() > deadline:
                return
            if current > limit:
                return
            if current > 1:
                smooth[current] = max_pf
            if idx >= len(prime_list):
                return
            p = prime_list[idx]
            # Try using this prime 0, 1, 2, ... times
            val = current
            while val <= limit:
                gen(idx + 1, val, max_pf)
                if val > limit // p:
                    break
                val *= p
        gen(0, 1, 1)
        return smooth
    
    # Generate primes via simple sieve
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    all_primes = sieve_primes(400)
    
    best_score = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    # Try increasing prime bounds
    for max_p in all_primes:
        if time.time() > deadline - 5:
            break
        # Get primes up to max_p
        primes = [p for p in all_primes if p <= max_p]
        smooth = generate_smooth(primes, TARGET)
        
        for a, pa in smooth.items():
            if time.time() > deadline - 3:
                break
            b = TARGET - a
            if b < 1:
                continue
            # Check if b is also smooth (much faster than factoring)
            if b in smooth:
                pb = smooth[b]
            else:
                pb = largest_prime_factor(b)
            score = max(pa, pb)
            if score < best_score:
                best_score = score
                best_a = a
                best_b = b
                if best_score == 2:
                    return [best_a, best_b]
        if best_score <= max_p:
            # Both a and b have all prime factors <= max_p
            break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
