import math

def construct():
    N = 1000000
    START = 10000
    
    # 1. Sieve of Smallest Prime Factors (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute Powerful Part for each number in range
    # We only need P[n] for n in [START, N]
    # But P[n+1] for n=N-1 is P[N], so we need up to N.
    P = [1] * (N + 1)
    
    for x in range(START, N + 1):
        temp = x
        p_part = 1
        while temp > 1:
            p = spf[temp]
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                p_part *= (p ** exp)
        P[x] = p_part

    # 3. Find n that maximizes P[n] * P[n+1]
    max_val = 0
    best_n = START
    
    for n in range(START, N):
        val = P[n] * P[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
            
    return best_n
