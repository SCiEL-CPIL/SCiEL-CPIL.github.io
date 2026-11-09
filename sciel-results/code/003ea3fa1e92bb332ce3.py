import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    N = 123457
    
    # Strategy: Find the optimal decomposition by searching for the smallest
    # possible value of max(P(a), P(b)) where a + b = N.
    
    # First, try to find a 5-smooth decomposition (both a and b are 5-smooth).
    # 5-smooth numbers have only prime factors 2, 3, 5.
    # If we can find such a pair, max(P(a), P(b)) <= 5, which is very good.
    
    # Generate all 5-smooth numbers up to N using a boolean array for fast lookup
    is_5smooth = [False] * (N + 1)
    is_5smooth[1] = True
    p2 = 1
    while p2 <= N:
        p3 = p2
        while p3 <= N:
            p5 = p3
            while p5 <= N:
                is_5smooth[p5] = True
                p5 *= 5
            p3 *= 3
        p2 *= 2
    
    # Linear scan for a where both a and N-a are 5-smooth
    for a in range(1, N):
        if is_5smooth[a] and is_5smooth[N - a]:
            return [a, N - a]
    
    # If no 5-smooth decomposition exists, try 7-smooth (add prime 7)
    is_7smooth = [False] * (N + 1)
    is_7smooth[1] = True
    p2 = 1
    while p2 <= N:
        p3 = p2
        while p3 <= N:
            p5 = p3
            while p5 <= N:
                p7 = p5
                while p7 <= N:
                    is_7smooth[p7] = True
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    
    for a in range(1, N):
        if is_7smooth[a] and is_7smooth[N - a]:
            return [a, N - a]
    
    # If no 7-smooth decomposition exists, try 11-smooth (add prime 11)
    is_11smooth = [False] * (N + 1)
    is_11smooth[1] = True
    p2 = 1
    while p2 <= N:
        p3 = p2
        while p3 <= N:
            p5 = p3
            while p5 <= N:
                p7 = p5
                while p7 <= N:
                    p11 = p7
                    while p11 <= N:
                        is_11smooth[p11] = True
                        p11 *= 11
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    
    for a in range(1, N):
        if is_11smooth[a] and is_11smooth[N - a]:
            return [a, N - a]
    
    # If no 11-smooth decomposition exists, use LPF sieve to find optimal
    # Compute largest prime factor for all numbers up to N using NumPy for speed
    lpf = np.ones(N + 1, dtype=np.int64)
    
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            lpf[i::i] = i
    
    # Now search for a that minimizes max(lpf[a], lpf[N-a])
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
