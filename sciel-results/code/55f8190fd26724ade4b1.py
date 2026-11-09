import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # we need values up to 1000001
    
    # Build SPF (smallest prime factor) sieve
    spf = list(range(MAX_VAL + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # powerful_part[m] = product of p^e for each prime power p^e || m with e >= 2
    powerful_part = [1] * (MAX_VAL + 1)
    
    for m in range(2, MAX_VAL + 1):
        x = m
        pp = 1
        while x > 1:
            p = spf[x]
            # Count exponent of p in x
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= (p ** e)
        powerful_part[m] = pp
    
    # Find n in [10000, 1000000] maximizing powerful_part[n] * powerful_part[n+1]
    best_n = 10000
    best_val = powerful_part[10000] * powerful_part[10001]
    
    for n in range(10001, N + 1):
        if time.time() > deadline - 2.0:
            break
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
