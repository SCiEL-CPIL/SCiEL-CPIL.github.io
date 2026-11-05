import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    Uses incremental smooth number generation.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    # We only need primes up to TARGET, but the solution will be found with small primes.
    # Sieve up to 1000 is sufficient for this specific target.
    primes = sieve(1000)
    
    # smooths contains all numbers whose prime factors are among primes processed so far
    smooths = {1}
    
    for idx, p in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate new smooth numbers that include prime p
        # Iterate over a copy of the current set to avoid modification during iteration issues
        # and to ensure we only multiply existing smooths (before p was added) by powers of p.
        # Actually, we want to generate ALL numbers using primes <= p.
        # The standard way is: for each s in current_smooths, multiply by p, p^2, ...
        # This generates new numbers that have p as a factor.
        # Numbers not having p are already in smooths.
        
        current_smooths = list(smooths)
        for s in current_smooths:
            val = s
            while val * p <= TARGET:
                val *= p
                smooths.add(val)
        
        # Check for pair: find a in smooths such that TARGET - a is also in smooths
        # We can restrict a <= TARGET/2 to avoid checking duplicates, but set lookup is fast.
        for a in smooths:
            b = TARGET - a
            if b > 0 and b in smooths:
                return [a, b]
    
    # Fallback if no solution found (should not happen)
    return [1, TARGET - 1]
