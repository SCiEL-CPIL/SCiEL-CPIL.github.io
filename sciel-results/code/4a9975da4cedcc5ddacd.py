import time
from math import isqrt, gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Determine optimal N
    # Primes 1 mod 4
    primes_1_4 = []
    p = 5
    while len(primes_1_4) < 20:
        # Check if p is prime
        is_prime = True
        for i in range(2, isqrt(p) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            primes_1_4.append(p)
        p += 2 if p == 5 else 2 # Step by 2, but ensure 1 mod 4?
        # Actually just iterate odd numbers
        # 5, 7(3 mod 4), 9(no), 11(3), 13(1)...
        # Better to just check all odd numbers
    
    # Regenerate properly
    primes_1_4 = []
    candidate = 5
    while len(primes_1_4) < 25:
        is_prime = True
        for i in range(2, isqrt(candidate) + 1):
            if candidate % i == 0:
                is_prime = False
                break
        if is_prime and candidate % 4 == 1:
            primes_1_4.append(candidate)
        candidate += 2
        
    # Knapsack to maximize product of (e_i+1)
    # Budget 18 (log10 N)
    # Items: (log10 p, p)
    # We want to assign exponents e_i
    # Greedy: sort by efficiency?
    # Actually, we determined that using small primes with e=1 is best, then incrementing smallest.
    # Let's just try configurations.
    
    best_N = 0
    best_count = 0
    best_factors = []
    
    # Try k from 10 to 15
    for k in range(10, 16):
        if k > len(primes_1_4):
            break
        # Use first k primes
        P = primes_1_4[:k]
        # Base exponents 1
        exponents = [1] * k
        log_sum = sum(__import__('math').log10(p) for p in P)
        
        if log_sum > 18:
            continue
            
        current_product = 2 ** k
        current_N = 1
        for p, e in zip(P, exponents):
            current_N *= p ** e
            
        # Try to increase exponents of smallest primes
        # Sort indices by log(p)
        indices = sorted(range(k), key=lambda i: __import__('math').log10(P[i]))
        
        for idx in indices:
            # Can we increase exponent?
            cost = __import__('math').log10(P[idx])
            if log_sum + cost <= 18:
                # Increase
                exponents[idx] += 1
                log_sum += cost
                current_N *= P[idx]
                # Update product: factor (e+1) becomes (e+2)
                # Old factor was (old_e + 1)
                # New factor is (old_e + 2)
                # Ratio is (old_e + 2) / (old_e + 1)
                old_e = exponents[idx] - 1
                current_product = current_product // (old_e + 1) * (old_e + 2)
            else:
                # Try next prime?
                pass
        
        # Calculate count
        # r_2(N) = 4 * current_product
        # Pairs = r_2(N) / 8
        count = current_product // 2
        
        if count > best_count:
            best_count = count
            best_N = current_N
            best_factors = [(p, e) for p, e in zip(P, exponents)]
            
    # 2. Generate pairs for best_N
    # Factor best_N is known in best_factors
    # Find representations x^2 + y^2 = best_N
    
    # Helper to find a^2 + b^2 = p for p 1 mod 4
    def sum_two_squares_prime(p):
        # Cornacchia's algorithm or brute force
        # p is small
        for a in range(1, isqrt(p) + 1):
            b2 = p - a*a
            b = isqrt(b2)
            if b*b == b2:
                return a, b
        return None, None

    # Combine representations
    # Start with 1 + 0i
    reps = [(1, 0)]
    
    for p, e in best_factors:
        a, b = sum_two_squares_prime(p)
        if a is None:
            continue
        # p = a^2 + b^2
        # We need to multiply current reps by (a+bi)^e
        # (x+yi)(a+bi) = (xa-yb) + i(xb+ya)
        # We only need real and imaginary parts squared
        new_reps = []
        for x, y in reps:
            # Compute (x+yi) * (a+bi)
            # But we need power e.
            # Iterate e times
            cx, cy = x, y
            for _ in range(e):
                nx = cx * a - cy * b
                ny = cx * b + cy * a
                cx, cy = nx, ny
            new_reps.append((cx, cy))
        reps = new_reps
        
    # Convert to pairs
    pairs = []
    seen = set()
    for x, y in reps:
        if x == 0 or y == 0:
            continue
        a = x * x
        b = y * y
        if a > b:
            a, b = b, a
        if a <= best_N and b <= best_N and a + b == best_N:
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append([a, b])
                
    # Sort pairs
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
