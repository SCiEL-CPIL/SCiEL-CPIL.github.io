import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Find primes 1 mod 4
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes_1_mod_4 = []
    n = 5
    while True:
        if n % 4 == 1 and is_prime(n):
            primes_1_mod_4.append(n)
        n += 4
        if n > 1000: break # Safety

    # Determine max k such that product of first k primes <= 10^18
    limit = 10**18
    N = 1
    k = 0
    for p in primes_1_mod_4:
        if N * p > limit:
            break
        N *= p
        k += 1
        
    if k == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    selected_primes = primes_1_mod_4[:k]
    
    # Find representation x^2 + y^2 = p
    reps = []
    for p in selected_primes:
        found = False
        for x in range(1, int(math.isqrt(p)) + 1):
            y2 = p - x*x
            y = int(math.isqrt(y2))
            if y*y == y2:
                reps.append((x, y))
                found = True
                break
        if not found:
            # Should not happen
            return {'N': 2, 'pairs': [[1, 1]]}
            
    # Generate representations
    current_reps = [(1, 0)]
    for x, y in reps:
        new_reps = []
        for u, v in current_reps:
            # Identity 1
            n1 = u*x - v*y
            n2 = u*y + v*x
            new_reps.append((n1, n2))
            
            # Identity 2
            n3 = u*x + v*y
            n4 = u*y - v*x
            new_reps.append((n3, n4))
        current_reps = new_reps
        
    pairs = set()
    for u, v in current_reps:
        a = u*u
        b = v*v
        if a > b:
            a, b = b, a
        pairs.add((a, b))
        
    pair_list = [[a, b] for a, b in pairs]
    
    return {'N': N, 'pairs': pair_list}
