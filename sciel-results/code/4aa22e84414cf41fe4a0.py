import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin

    N = 1000000
    best_n = 10000

    pp = np.ones(N + 2, dtype=np.int64)

    limit = int(math.isqrt(N)) + 1
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            continue

        pe = p2
        while pe <= N + 1:
            pe_next = pe * p
            if pe_next > N + 1:
                pp[pe::pe] *= pe
            else:
                # k not divisible by p: vectorized mask approach
                max_k = (N + 1) // pe
                if max_k > 0:
                    k_vals = np.arange(1, max_k + 1)
                    mask = (k_vals % p) != 0
                    indices = k_vals[mask] * pe
                    pp[indices] *= pe
            pe = pe_next
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    max_prod = 0
    for n in range(10000, N + 1):
        prod = int(pp[n]) * int(pp[n+1])
        if prod > max_prod:
            max_prod = prod
            best_n = n
        if time.time() > deadline:
            break

    return best_n
