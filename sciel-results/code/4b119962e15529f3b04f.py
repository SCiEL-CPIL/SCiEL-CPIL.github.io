import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s buffer before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_N = N_MAX + 1  # we need n and n+1, so up to 1000001
    
    # powerful_part[i] will hold the powerful part of i
    powerful_part = [1] * (MAX_N + 1)
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each number, compute its powerful part by factoring
    # For each prime p, for each power p^e with e >= 2,
    # multiply p^e into powerful_part[m] for all m where p^e divides m
    
    # Collect all primes
    primes = []
    for i in range(2, MAX_N + 1):
        if spf[i] == i:
            primes.append(i)
    
    # For each prime, for each power p^e with e >= 2,
    # multiply p^e into powerful_part[m] for all m divisible by p^e
    for p in primes:
        p2 = p * p
        if p2 > MAX_N:
            break
        pe = p2
        while pe <= MAX_N:
            for m in range(pe, MAX_N + 1, pe):
                powerful_part[m] *= pe
            pe *= p
    
    # Find the best n in [N_MIN, N_MAX] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = N_MIN
    best_val = powerful_part[N_MIN] * powerful_part[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
