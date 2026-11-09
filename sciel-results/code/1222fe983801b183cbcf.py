def construct():
    N = 123457
    
    # Generate 5-smooth numbers up to N
    # 5-smooth numbers are of the form 2^a * 3^b * 5^c
    smooth_set = set()
    
    # Generate all 5-smooth numbers
    i = 1
    while i <= N:
        j = i
        while j <= N:
            k = j
            while k <= N:
                smooth_set.add(k)
                k *= 5
            j *= 3
        i *= 2
    
    # Check if both a and N-a are 5-smooth
    # If so, max(P(a), P(b)) <= 5, which is optimal since we need at least 2
    # Actually, if both are 5-smooth, their largest prime factors are at most 5
    # But we can do even better: check for 3-smooth (Hamming numbers with primes 2,3)
    # If both a and N-a are 3-smooth, max(P(a), P(b)) <= 3
    
    # First check for 3-smooth (primes 2, 3 only)
    smooth3_set = set()
    i = 1
    while i <= N:
        j = i
        while j <= N:
            smooth3_set.add(j)
            j *= 3
        i *= 2
    
    for a in range(1, N):
        b = N - a
        if a in smooth3_set and b in smooth3_set:
            # Both are 3-smooth, so max(P(a), P(b)) <= 3
            # Since N is odd, one of a,b is even and one is odd
            # The odd one must be a power of 3
            return [a, b]
    
    # If no 3-smooth pair found, check for 5-smooth
    for a in range(1, N):
        b = N - a
        if a in smooth_set and b in smooth_set:
            # Both are 5-smooth, so max(P(a), P(b)) <= 5
            return [a, b]
    
    # Fallback: compute largest prime factor and find optimal decomposition
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i

    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1

    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            
            if best_max_pf <= 5:
                break

    return [best_a, best_b]
