import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate primes p = 1 mod 4
    # We need the first ~20 primes to be safe, but product grows fast.
    # 5 * 13 * ... 
    # Let's sieve up to 1000 to get enough primes.
    limit_sieve = 1000
    is_prime = [True] * (limit_sieve + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit_sieve**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_sieve + 1, i):
                is_prime[j] = False
    
    primes_1_4 = [p for p in range(5, limit_sieve + 1) if is_prime[p] and p % 4 == 1]
    
    # 2. Determine max t such that product of first t primes <= 10^18
    M = 1
    selected_primes = []
    for p in primes_1_4:
        if M * p > 10**18:
            break
        M *= p
        selected_primes.append(p)
        
    # If we didn't reach 10^18, we could potentially multiply by a square s^2
    # to get closer to 10^18? No, N just needs to be <= 10^18.
    # The number of pairs depends only on the prime factorization of the 'core' M.
    # Adding a square factor s^2 to N (making N = s^2 * M) scales the representations:
    # if x^2 + y^2 = M, then (sx)^2 + (sy)^2 = s^2 M.
    # The number of pairs is the same.
    # So we can just use N = M.
    
    # 3. Find representations p = a^2 + b^2 for each selected prime
    # For p = 1 mod 4, there is a unique representation (up to order/sign).
    # We can find it by checking x from 1 to isqrt(p)
    prime_reps = []
    for p in selected_primes:
        found = False
        x = 1
        while x * x < p:
            y2 = p - x * x
            y = int(math.isqrt(y2))
            if y * y == y2:
                prime_reps.append((x, y))
                found = True
                break
            x += 1
        if not found:
            # Should not happen for p=1 mod 4
            prime_reps.append((1, 0)) # Fallback, though incorrect
        if time.time() > deadline - 10:
            break

    # 4. Generate all representations of M as sum of two squares
    # Start with (1, 0) which is 1^2 + 0^2 = 1
    # Multiply by (a+ib) or (a-ib) for each prime factor
    # (x1 + i y1) * (x2 + i y2) = (x1 x2 - y1 y2) + i (x1 y2 + y1 x2)
    # We track (real, imag)
    
    reps = [(1, 0)]
    
    for a, b in prime_reps:
        new_reps = []
        for u, v in reps:
            # Option 1: multiply by (a + ib)
            # (u + iv)(a + ib) = (ua - vb) + i(ub + va)
            r1 = u * a - v * b
            i1 = u * b + v * a
            new_reps.append((r1, i1))
            
            # Option 2: multiply by (a - ib) (conjugate)
            # (u + iv)(a - ib) = (ua + vb) + i(va - ub)
            r2 = u * a + v * b
            i2 = v * a - u * b
            new_reps.append((r2, i2))
        
        reps = new_reps
        
        if time.time() > deadline - 5:
            break

    # 5. Convert to powerful number pairs
    # Each rep (u, v) gives a solution u^2 + v^2 = M.
    # a = u^2, b = v^2.
    # a and b are squares, so they are powerful.
    # We need distinct pairs a <= b.
    
    pairs_set = set()
    for u, v in reps:
        a = u * u
        b = v * v
        if a > b:
            a, b = b, a
        # Ensure a,b are positive? 
        # Powerful numbers usually defined for positive integers. 
        # The problem says "1 counts as powerful". 0 is not usually considered in this context unless specified.
        # "Integers up to 10^18". Sums of two powerful numbers.
        # If u=0, a=0. Is 0 powerful? 
        # Standard definition: n is powerful if for every prime p dividing n, p^2 divides n.
        # 0 is divisible by all primes. 0 is not typically called powerful in this context (sums of two squares usually implies non-negative, but powerful numbers are positive).
        # Let's assume a,b >= 1.
        if a >= 1 and b >= 1:
            pairs_set.add((a, b))
            
    pairs_list = [list(p) for p in pairs_set]
    
    if not pairs_list:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Sort pairs for consistency (optional, but good)
    pairs_list.sort()
    
    N = M
    
    return {'N': N, 'pairs': pairs_list}
