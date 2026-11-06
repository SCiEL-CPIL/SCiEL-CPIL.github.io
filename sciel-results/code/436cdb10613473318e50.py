import math

def construct():
    # Generate primes p == 1 mod 4
    # We need the smallest ones to maximize the count k
    primes_1mod4 = []
    n = 5
    # We only need to find enough primes such that their product exceeds 10^18
    # Estimate: log10(10^18) = 18.
    # Smallest prime is 5 (log10 ~ 0.7).
    # Next 13 (1.1), 17 (1.23)...
    # We can safely generate primes up to, say, 200, which gives ~15-20 primes.
    while len(primes_1mod4) < 25:
        if n % 4 == 1:
            # Primality check
            is_prime = True
            if n > 1:
                limit = int(math.isqrt(n))
                for i in range(3, limit + 1, 2):
                    if n % i == 0:
                        is_prime = False
                        break
            if is_prime:
                primes_1mod4.append(n)
        n += 2

    # Select the maximum number of smallest primes such that product <= 10^18
    N = 1
    selected_primes = []
    for p in primes_1mod4:
        if N * p <= 10**18:
            N *= p
            selected_primes.append(p)
        else:
            break
            
    # If N is 1, we need at least one prime. 
    # Smallest prime 1 mod 4 is 5.
    # 5 <= 10^18, so N will be at least 5.
    
    # Find representation p = a^2 + b^2 for each prime
    # For p == 1 mod 4, such representation exists and is unique up to order/sign.
    prime_reps = []
    for p in selected_primes:
        found = False
        # a^2 + b^2 = p => a^2 <= p/2
        limit_a = int(math.isqrt(p // 2))
        for a in range(1, limit_a + 1):
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                # Ensure a <= b to keep it canonical? 
                # The multiplication formula works regardless, but we want unique reps at the end.
                # The final filtering handles uniqueness.
                prime_reps.append((a, b))
                found = True
                break
        if not found:
            # Should not happen for p == 1 mod 4
            prime_reps.append((1, 0)) # Fallback, though 1^2+0^2=1, not p.
            # Actually, if p is prime 1 mod 4, it must be sum of two squares.
            # The loop must find it.
            
    # Combine representations using Gaussian Integer multiplication
    # Start with (1, 0) representing 1 + 0i
    reps = [(1, 0)]
    
    for a, b in prime_reps:
        new_reps = []
        for u, v in reps:
            # (u + iv)(a + ib) = (ua - vb) + i(ub + va)
            u1 = u * a - v * b
            v1 = u * b + v * a
            new_reps.append((u1, v1))
            
            # (u + iv)(a - ib) = (ua + vb) + i(va - ub)
            u2 = u * a + v * b
            v2 = v * a - u * b
            new_reps.append((u2, v2))
        reps = new_reps
        
    # Filter and create pairs
    pairs = []
    seen = set()
    for u, v in reps:
        # We want positive u, v
        if u <= 0 or v <= 0:
            continue
        
        # Ensure distinct pairs a <= b
        # a = u^2, b = v^2
        # We want u^2 <= v^2 => |u| <= |v|
        # Since u,v > 0, u <= v
        if u > v:
            # Swap to maintain u <= v
            u, v = v, u
            
        # Check if u^2 + v^2 == N (Safety)
        if u * u + v * v != N:
            continue
            
        a_val = u * u
        b_val = v * v
        
        # Since u <= v, a_val <= b_val
        pair = (a_val, b_val)
        
        if pair not in seen:
            seen.add(pair)
            pairs.append([a_val, b_val])
            
    # The problem requires distinct pairs.
    # My logic ensures distinct (a,b) values.
    
    return {'N': N, 'pairs': pairs}
