import math
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N)
    
    best_set = []
    
    # Try to find the largest k such that the set of numbers coprime to P(k) has size >= k
    # P(k) is product of primes <= k
    # We can just generate the set for a given k
    
    # Estimate max k
    # Product decreases as k increases.
    # Start from high k and go down?
    # Or start from low k and go up?
    # The size of the set DECREASES as k increases (more constraints).
    # So we want the largest k such that Size(k) >= k.
    # Size(k) is decreasing. k is increasing.
    # So there is a unique crossing point.
    
    # Let's iterate k from 1 to N
    # But calculating the set is expensive.
    
    # Optimization: 
    # The set for k+1 is a subset of the set for k.
    # So we can maintain the set.
    
    current_set = set(range(N + 1))
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Find the next prime <= k?
        # No, we need to add constraint for prime p if p <= k.
        # But k is the size, not the bound for primes.
        # The condition is for primes p <= |T|.
        # If |T| = k, we need constraints for p <= k.
        
        # This is tricky because |T| depends on the constraints.
        
        # Alternative:
        # Iterate possible sizes k from N down to 1.
        # For each k, check if there exists an admissible set of size k.
        # This is hard.
        
        # Let's stick to the "coprime to P(k)" strategy.
        # We need to find k such that Count(Coprime(P(k))) >= k.
        
        # Let's just calculate for a few k values.
        
        pass
        
    # Fallback to seed logic if the above is too slow or complex
    # But we can implement the seed logic which is known to work.
    
    points = [1] + [p for p in primes if p * p > N]
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    return points
