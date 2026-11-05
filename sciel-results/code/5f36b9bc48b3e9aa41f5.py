import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    Uses incremental smooth number generation: add one prime at a time,
    generating only the new smooth numbers that include that prime.
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

    primes = sieve(TARGET)
    
    # Incremental smooth number generation
    # smooths contains all numbers whose prime factors are among primes processed so far
    smooths = {1}
    best_pair = None
    
    for idx, q in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate new smooth numbers that include prime q
        # For each existing smooth number s, multiply by q, q^2, q^3, ...
        new_smooths = []
        for s in list(smooths):
            val = s
            while val * q <= TARGET:
                val *= q
                new_smooths.append(val)
        
        # Add new smooths to the set
        for ns in new_smooths:
            smooths.add(ns)
        
        # Check for pair: only need to check newly added smooths
        # because old smooths were already checked in previous iterations
        for a in new_smooths:
            b = TARGET - a
            if b > 0 and b in smooths:
                # Since we're adding primes in increasing order, the first pair found
                # has the minimum possible max(P(a), P(b))
                return [a, b]
    
    if best_pair is not None:
        return best_pair
    return [1, TARGET - 1]
