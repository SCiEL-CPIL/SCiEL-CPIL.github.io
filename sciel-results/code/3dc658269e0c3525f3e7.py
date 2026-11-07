import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 82  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Fast multiplicative sigma sieve: O(n log log n)
    # sigma[1] = 1
    # For each prime p, for each multiple m of p, update sigma[m] by multiplying
    # by (1 + p + p^2 + ... + p^e) / (1 + p + ... + p^(e-1)) where p^e || m
    
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    
    # smallest prime factor sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma using the multiplicative property
    # For each n, factorize using spf and compute sigma
    # sigma[n] = product over p^e || n of (p^(e+1)-1)/(p-1)
    
    # More efficient: use the recurrence
    # If n = p^e * m where gcd(m,p)=1, then sigma[n] = sigma[m] * (p^(e+1)-1)/(p-1)
    # But we need to be careful with the division.
    
    # Alternative: use the additive sieve but optimized
    # sigma[n] = sum_{d|n} d
    # We can compute this as: for d from 1 to n, add d to sigma[d], sigma[2d], sigma[3d], ...
    
    # The additive sieve is O(n log n) which for n=10^7 is about 10^7 * ln(10^7) ≈ 1.6 * 10^8
    # That's feasible in C but might be slow in Python. Let's try the multiplicative approach.
    
    # Actually, let's use a different approach:
    # For each n, find its prime factorization using spf, then compute sigma.
    # This is O(n * number_of_prime_factors) which is fast.
    
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        p = spf[n]
        m = n // p
        # Count power of p in n
        e = 0
        while m % p == 0:
            e += 1
            m //= p
        # n = p^(e+1) * m
        # sigma[n] = sigma[m] * (p^(e+2)-1)/(p-1) / [(p^(e+1)-1)/(p-1)] ... no
        # Actually: sigma[p^(e+1) * m] = sigma[m] * sigma[p^(e+1)] = sigma[m] * (p^(e+2)-1)/(p-1)
        # Wait, that's not right either. sigma is multiplicative:
        # sigma[p^k * m] = sigma[p^k] * sigma[m] when gcd(p^k, m) = 1
        # sigma[p^k] = 1 + p + p^2 + ... + p^k = (p^(k+1)-1)/(p-1)
        
        # Let me redo: n = p^(e+1) * m where gcd(m,p)=1
        # sigma[n] = sigma[m] * (1 + p + p^2 + ... + p^(e+1))
        # = sigma[m] * (p^(e+2) - 1) / (p - 1)
        
        # But I need to compute this without floating point.
        # (p^(e+2) - 1) / (p - 1) = 1 + p + p^2 + ... + p^(e+1)
        
        # For small e, just compute the sum directly.
        # For large e, use the formula with integer division.
        
        if e == 0:
            # n = p * m, gcd(m,p)=1
            # sigma[n] = sigma[m] * (1 + p)
            sigma[n] = sigma[m] * (1 + p)
        else:
            # n = p^(e+1) * m
            # Compute p^(e+2) - 1, then divide by p - 1
            # For e+1 up to about 23 (2^23 > 10^7), this is fine
            pe = p ** (e + 1)
            # sigma[p^(e+1)] = (p^(e+2)-1)/(p-1) = (pe * p - 1) // (p - 1)
            sigma_pe = (pe * p - 1) // (p - 1)
            sigma[n] = sigma[m] * sigma_pe
    
    # Now group numbers by sigma value
    sigma_to_numbers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_to_numbers[sigma[n]].append(n)
    
    best_error = float('inf')
    best_a = None
    best_b = None
    
    # Search for a/b close to 3/2
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # Let d = 2a - 3b, then a = (3b + d)/2
    # For fixed d, error = |d|/(2b), so we want large b
    
    # For each d (starting from 1), find b such that:
    # 1. a = (3b+d)/2 is integer (b and d same parity)
    # 2. 1 <= a <= MAX_N, 1 <= b <= MAX_N
    # 3. sigma[a] == sigma[b]
    
    # Check d values in order of |d|
    max_d = 200  # Check d values up to 200
    
    for d in range(1, max_d + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # b and d must have same parity
        # a = (3b + d) / 2
        # 1 <= (3b+d)/2 <= MAX_N => 2-d <= 3b <= 2*MAX_N-d
        # b >= ceil((2-d)/3), b <= floor((2*MAX_N-d)/3)
        
        b_min = max(1, math.ceil((2 - d) / 3))
        b_max = min(MAX_N, math.floor((2 * MAX_N - d) / 3))
        
        # Adjust b_min to correct parity (b and d same parity)
        if (b_min + d) % 2 != 0:
            b_min += 1
        
        # For this d, the minimum possible error is |d|/(2*b_max)
        # If that's already >= best_error, skip this d
        if d / (2 * b_max) >= best_error:
            break  # All larger d will be worse
        
        # Now search for b in [b_min, b_max] with correct parity
        # where sigma[(3b+d)//2] == sigma[b]
        
        # Iterate b from b_max down to b_min (largest b first for smallest error)
        # But this could be millions of iterations.
        
        # Key optimization: for each b, a = (3b+d)//2 is determined.
        # We need sigma[a] == sigma[b].
        # We can check this in O(1) if we have the sigma array.
        
        # The issue is iterating over potentially millions of b values.
        # Let's limit the number of checks per d.
        
        # Actually, let's think differently. For a given d, a = (3b+d)/2.
        # So b = (2a - d)/3. For a to give integer b, 2a - d must be divisible by 3.
        # And b must be in [1, MAX_N].
        
        # Alternative: iterate over a values and check if b = (2a-d)/3 is valid.
        # a ranges from ceil((2+d)/3) to MAX_N... wait let me redo.
        # b = (2a-d)/3, need 1 <= b <= MAX_N
        # 3 <= 2a-d <= 3*MAX_N => (3+d)/2 <= a <= (3*MAX_N+d)/2
        # a_min = ceil((3+d)/2), a_max = min(MAX_N, floor((3*MAX_N+d)/2))
        
        # Hmm, both approaches are similar in complexity.
        
        # Let's just iterate b in descending order but with a time check.
        # For d=1, b_max ≈ 6.67M. That's a lot.
        
        # Better idea: for each d, instead of iterating all b, 
        # iterate over the sigma groups and check if there's a pair.
        
        # Actually, the most efficient approach:
        # For each d, the mapping b -> a = (3b+d)/2 is a bijection between
        # valid b values and valid a values.
        # So we need to find b in the valid range such that sigma[(3b+d)//2] = sigma[b].
        
        # Let's just do a direct scan but with early termination.
        # For d=1, we check b from ~6.67M down. The first match gives the best error for d=1.
        
        checked = 0
        max_checks = 2000000  # Limit checks per d
        
        for b in range(b_max, b_min - 1, -2 if d % 2 == 0 else -2):
            # Wait, step should be -2 to maintain parity
            pass
        
        # Let me redo the loop properly
        step = -2  # maintain parity
        b = b_max
        if (b + d) % 2 != 0:
            b -= 1
        
        while b >= b_min:
            if time.time() - start_time > DEADLINE:
                break
            
            a = (3 * b + d) // 2
            
            if a < 1 or a > MAX_N:
                b += step
                continue
            
            if sigma[a] == sigma[b]:
                error = d / (2 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                # Found a match for this d, can stop (largest b gives smallest error for this d)
                break
            
            b += step
            checked += 1
            if checked >= max_checks:
                break
    
    if best_a is None:
        return [int(11), int(6)]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
