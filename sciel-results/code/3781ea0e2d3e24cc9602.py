import time

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    # Sieve to compute largest prime factor for all numbers up to TARGET
    lpf = [0] * (TARGET + 1)
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i
    
    # lpf[1] = 1 by definition
    lpf[1] = 1
    
    # Find the pair (a, b) with a + b = TARGET minimizing max(lpf[a], lpf[b])
    best_max = TARGET  # upper bound
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
    
    return [best_a, best_b]
