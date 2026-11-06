import math

def construct():
    # Generate primes p ≡ 1 (mod 4) using a simple sieve approach for efficiency
    # We need the smallest such primes to maximize k where prod(p_i) <= 10^18
    
    # Let's generate primes up to some limit using sieve
    limit = 500  # Should be more than enough for the first 20+ primes ≡ 1 mod 4
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes_1mod4 = []
    for n in range(5, limit + 1, 4):  # Only check n ≡ 1 mod 4
        if is_prime[n]:
            primes_1mod4.append(n)
    
    # Select largest k such that product <= 10^18
    best_k = 0
    prod_val = 1
    for i, p in enumerate(primes_1mod4):
        if prod_val * p <= 10**18:
            prod_val *= p
            best_k = i + 1
        else:
            break
    
    N = prod_val
    selected_primes = primes_1mod4[:best_k]
    
    # For each prime p ≡ 1 mod 4, find the representation p = a^2 + b^2
    # Using Cornacchia's algorithm or simple search
    prime_reps = []
    for p in selected_primes:
        found = False
        # a^2 + b^2 = p, with 0 < a < b
        for a in range(1, int(math.isqrt(p // 2)) + 1):
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                prime_reps.append((a, b))
                found = True
                break
        if not found:
            # This shouldn't happen for primes ≡ 1 mod 4
            # But just in case, use a fallback
            prime_reps.append((1, 1))
    
    # Combine representations using Gaussian integer multiplication
    # Start with (1, 0) representing 1
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
    
    # Filter and create valid pairs
    pairs = []
    seen = set()
    for u, v in reps:
        # We need u^2 + v^2 = N
        # And u, v > 0
        # And u < v (to ensure a <= b and distinct)
        if u <= 0 or v <= 0:
            continue
        if u >= v:
            continue
        a_val = u * u
        b_val = v * v
        if a_val + b_val != N:
            continue  # Safety check
            
        pair = (a_val, b_val)
        if pair not in seen:
            seen.add(pair)
            pairs.append([a_val, b_val])
    
    return {'N': N, 'pairs': pairs}
