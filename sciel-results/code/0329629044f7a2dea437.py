# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # Need LPF up to N_MAX+1
    
    # Sieve to compute largest prime factor for each number up to limit
    lpf = [0] * (limit + 1)
    
    # Sieve of Eratosthenes style: for each prime, update its multiples
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i  # i is the largest prime factor seen so far for j
    
    # Find n that minimizes max(LPF[n], LPF[n+1])
    best_n = N_MIN
    best_score = lpf[N_MIN] if lpf[N_MIN] >= lpf[N_MIN + 1] else lpf[N_MIN + 1]
    
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() > deadline:
            break
        score = lpf[n] if lpf[n] >= lpf[n + 1] else lpf[n + 1]
        if score < best_score:
            best_score = score
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
