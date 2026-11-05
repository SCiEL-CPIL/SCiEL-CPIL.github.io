import time

def construct():
    TARGET = 123457
    start_time = time.time()
    
    # Compute LPF (largest prime factor) for all numbers up to TARGET
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i
    
    # Find a that minimizes max(lpf[a], lpf[TARGET-a])
    best_max_p = TARGET
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET // 2 + 1):
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
    
    return [best_a, best_b]
