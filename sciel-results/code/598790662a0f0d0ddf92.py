import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n, with P(1)=1.
    
    Strategy: Incrementally build the set of K-smooth numbers as we add primes.
    For each K, check if any a in the set has TARGET-a also in the set.
    The first K for which such a pair exists gives the minimal max(P(a), P(b)).
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Incrementally build smooth set
    smooth_set = {1}
    
    for p in primes:
        if time.time() - start_time > DEADLINE:
            return [1, TARGET - 1]
        
        # Add all numbers formed by multiplying existing smooth numbers by powers of p
        new_smooths = set()
        for s in smooth_set:
            v = s * p
            while v <= TARGET:
                new_smooths.add(v)
                v *= p
        smooth_set.update(new_smooths)
        
        # Check for a pair a + b = TARGET with both in smooth_set
        for a in smooth_set:
            b = TARGET - a
            if b > 0 and b in smooth_set:
                return [a, b]
    
    # Fallback
    return [1, TARGET - 1]
