import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    N = 1000000
    LOW = 10000

    # Build smallest prime factor table up to N+1
    spf = list(range(N + 2))
    for i in range(2, int((N + 1) ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 2, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute largest prime factor for all numbers 0..N+1
    lpf_arr = [1] * (N + 2)
    for i in range(2, N + 2):
        p = spf[i]
        n = i
        while n % p == 0:
            n //= p
        if n == 1:
            lpf_arr[i] = p
        else:
            lpf_arr[i] = max(p, lpf_arr[n])

    # Find n in [LOW, N] that minimizes max(lpf(n), lpf(n+1))
    best_n = LOW
    best_val = max(lpf_arr[LOW], lpf_arr[LOW + 1])

    for n in range(LOW + 1, N + 1):
        if time.time() > deadline:
            break
        v = max(lpf_arr[n], lpf_arr[n + 1])
        if v < best_val:
            best_val = v
            best_n = n

    return best_n
