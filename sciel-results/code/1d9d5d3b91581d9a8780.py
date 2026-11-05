import time

def construct():
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85
    
    # Precompute largest prime factor (LPF) for all numbers up to TARGET
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i  # Since we iterate primes in order, last assignment is largest
    
    # Scan all valid splits
    best_max_p = TARGET + 1
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
    
    return [best_a, best_b]
