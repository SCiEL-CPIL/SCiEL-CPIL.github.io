import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    N = 123457
    
    # Strategy 1: Try to find a 5-smooth decomposition
    # 5-smooth numbers are of the form 2^a * 3^b * 5^c
    # If both a and N-a are 5-smooth, then max(P(a), P(N-a)) <= 5
    # This is optimal since for any n >= 2, P(n) >= 2, and P(1) = 1
    # But we need a+b=N with both positive, so minimum possible max is at least 2
    # If we can achieve max <= 5, that's very good.
    
    # Generate all 5-smooth numbers up to N
    smooth_set = set()
    p2 = 1
    while p2 <= N:
        p3 = p2
        while p3 <= N:
            p5 = p3
            while p5 <= N:
                smooth_set.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    
    # Check if there exists a in smooth_set such that N-a is also in smooth_set
    best_a_smooth = None
    for a in smooth_set:
        if a < N:  # b = N - a must be positive
            b = N - a
            if b in smooth_set:
                # Found a 5-smooth decomposition
                # Verify: both are positive and sum to N
                if a >= 1 and b >= 1:
                    best_a_smooth = a
                    break
    
    if best_a_smooth is not None:
        return [best_a_smooth, N - best_a_smooth]
    
    # Strategy 2: Fall back to LPF sieve
    lpf = [1] * (N + 1)
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_a = 1
    best_val = max(lpf[1], lpf[N - 1])
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    return [best_a, N - best_a]
