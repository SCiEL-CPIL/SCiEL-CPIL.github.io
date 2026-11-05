import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    N = 1000000
    P = [1] * (N + 1)  # P[i] = powerful part of i
    
    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # For each prime, multiply p^k into P[m] for appropriate multiples
    for p in primes:
        if time.time() > deadline:
            break
        power = p * p  # p^2
        while power <= N:
            next_power = power * p  # p^(k+1)
            # For each multiple m of power, if p^(k+1) does not divide m, multiply P[m] by power
            for m in range(power, N + 1, power):
                if m % next_power != 0:
                    P[m] *= power
            power = next_power
            if power > N:
                break
    
    # Find n in [10000, 1000000] that maximizes P[n] * P[n+1]
    best_val = 0
    best_n = 10000
    for n in range(10000, N):  # n+1 must be <= N
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
