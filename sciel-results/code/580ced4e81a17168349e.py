import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def phi(x):
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def batch_phi(start, count):
        if count <= 0:
            return []
        end = start + count - 1
        phi_vals = list(range(start, end + 1))
        rem = list(range(start, end + 1))
        for p in primes:
            if p > end:
                break
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            for m in range(first, end + 1, p):
                idx = m - start
                if rem[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        for i in range(count):
            if rem[i] > 1:
                phi_vals[i] -= phi_vals[i] // rem[i]
        return phi_vals

    best_start = 1
    best_len = 1

    if time.time() > deadline:
        return {'start': best_start, 'length': best_len}

    random.seed(777)
    candidates = set()

    for base in [10**10, 3*10**10, 10**11, 2*10**11, 4*10**11, 6*10**11, 8*10**11, 9*10**11]:
        for _ in range(100):
            s = base + random.randint(0, 10**6)
            if 1 <= s and s + 200 <= 10**12:
                candidates.add(s)

    for _ in range(500):
        s = random.randint(10**9, 10**12 - 300)
        candidates.add(s)

    for _ in range(300):
        s = random.randint(7*10**11, 10**12 - 300)
        candidates.add(s)

    candidates = sorted(candidates)

    for start in candidates:
        if time.time() > deadline - 3.0:
            break
        if start < 1 or start + 200 > 10**12:
            continue

        phi_vals = batch_phi(start, 200)

        seen = set()
        k = 0
        for i in range(200):
            v = phi_vals[i]
            if v in seen:
                break
            seen.add(v)
            k += 1

        if k > best_len:
            best_len = k
            best_start = start

    if best_len > 1:
        check = batch_phi(best_start, best_len)
        if len(set(check)) != best_len:
            return {'start': 1, 'length': 1}

    return {'start': best_start, 'length': best_len}
