import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 20s margin
    
    # Primes 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89]
    
    # Exponents determined by optimization
    # 5^2, 13^2, 17^2, 29^1, 37^1, 41^1, 53^1, 61^1, 73^1, 89^1
    exponents = [2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    
    N = 1
    for p, e in zip(primes, exponents):
        N *= p ** e
        
    # Check if N <= 10^18
    if N > 10**18:
        # Fallback to smaller N
        N = 5 * 13 * 17 * 29 * 37 * 41 * 53 * 61 * 73 * 89 * 97
        exponents = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]

    # Generate all pairs (x, y) such that x^2 + y^2 = N
    # Using recursive combination of prime factors
    
    def get_prime_pair(p):
        # Find u, v such that u^2 + v^2 = p
        # p is prime 1 mod 4
        for u in range(1, int(math.isqrt(p)) + 1):
            v2 = p - u * u
            v = int(math.isqrt(v2))
            if v * v == v2:
                return (u, v)
        return None

    # Base cases for each prime power
    # For p^1, we have (u, v)
    # For p^k, we can combine (u, v) k times?
    # Actually, we just need to generate all representations of N.
    # We can use the formula:
    # If x^2 + y^2 = A and u^2 + v^2 = B, then
    # (xu - yv)^2 + (xv + yu)^2 = AB
    # (xu + yv)^2 + (xv - yu)^2 = AB
    
    reps = [(1, 0)]
    
    for p, e in zip(primes, exponents):
        if e == 0:
            continue
        u, v = get_prime_pair(p)
        # We need to apply the factor p^e
        # This is equivalent to multiplying by (u+iv)^e in Gaussian integers
        # But we can just iteratively combine
        new_reps = []
        for _ in range(e):
            temp_reps = []
            for x, y in new_reps if new_reps else reps:
                # Combine (x,y) with (u,v)
                x1 = x * u - y * v
                y1 = x * v + y * u
                x2 = x * u + y * v
                y2 = x * v - y * u
                temp_reps.append((abs(x1), abs(y1)))
                temp_reps.append((abs(x2), abs(y2)))
            reps = temp_reps
        # Deduplicate and normalize
        unique_reps = set()
        for x, y in reps:
            if x > y:
                x, y = y, x
            unique_reps.add((x, y))
        reps = list(unique_reps)
        
        if time.time() > deadline:
            break

    # Filter pairs where x, y > 0 and x != y (since N is not a square)
    # And x^2 + y^2 = N
    pairs = []
    for x, y in reps:
        if x == 0 or y == 0:
            continue
        if x * x + y * y != N:
            continue
        a = x * x
        b = y * y
        if a <= b:
            pairs.append([a, b])
            
    # Deduplicate pairs
    pairs = list(set(map(tuple, pairs)))
    pairs.sort()
    
    return {'N': N, 'pairs': pairs}
