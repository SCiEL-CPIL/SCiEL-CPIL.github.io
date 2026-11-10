import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    SIZE = N_MAX + 2  # To check n+1 up to 1000001
    
    # Sieve to compute Largest Prime Factor (LPF) for each number up to SIZE-1
    # lpf[i] = largest prime factor of i
    lpf = [0] * SIZE
    for i in range(2, SIZE):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, SIZE, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    # Now find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
