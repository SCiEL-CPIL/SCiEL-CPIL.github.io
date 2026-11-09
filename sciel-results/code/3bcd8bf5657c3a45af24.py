import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    N = 123457
    
    # Sieve to compute largest prime factor for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (N + 1)
    
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Scan all decompositions a from 1 to N//2 (symmetry: (a,b) and (b,a) give same max)
    for a in range(1, N // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            # Early termination: if max_pf is 2, that's optimal (both must be powers of 2, 
            # but N is odd so impossible). If max_pf is 3, check if achievable.
            # For N=123457 (odd), one of a,b is even, one is odd.
            # If max_pf <= 2, both must be powers of 2, sum would be power of 2 or 0 mod 3, etc.
            # 123457 is odd, so cannot be sum of two powers of 2 (which would be even or have specific form).
            # The theoretical minimum for odd N is at least 3 (one number is 1*3^k or similar, other is power of 2).
            # If we find max_pf == 3, we can stop since we can't do better for odd N.
            if m <= 3:
                break
                
    return [best_a, best_b]
