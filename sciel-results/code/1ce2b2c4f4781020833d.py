import time
import math

def construct():
    start = time.time()
    deadline = start + 110

    P_MIN = 1000
    P_MAX = 100_000
    Q_MIN = P_MAX
    Q_MAX = 1_000_000

    def sieve(limit):
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return is_prime

    is_prime_arr = sieve(Q_MAX)
    primes_q = [i for i in range(Q_MIN, Q_MAX + 1) if is_prime_arr[i]]
    primes_p = [i for i in range(2, P_MAX + 1) if is_prime_arr[i]]

    size = P_MAX - P_MIN + 1
    first_bad = [201] * size

    def tonelli_shanks(n, p):
        if p == 2:
            return n
        if pow(n, (p - 1) // 2, p) != 1:
            return None
        if p % 4 == 3:
            return pow(n, (p + 1) // 4, p)
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1
        m = s
        c = pow(z, q, p)
        t = pow(n, q, p)
        r = pow(n, (q + 1) // 2, p)
        while t != 1:
            i = 0
            tmp = t
            while tmp != 1:
                tmp = (tmp * tmp) % p
                i += 1
            b = pow(c, 1 << (m - i - 1), p)
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p
        return r

    for q in primes_q:
        if time.time() > deadline:
            break
        half_q = (q - 1) // 2
        for i in range(1, 201):
            a = (-i) % q
            if pow(a, half_q, q) != 1:
                continue
            x1 = tonelli_shanks(a, q)
            if x1 is None:
                continue
            x2 = (-x1) % q
            for x in [x1, x2]:
                if x < P_MIN:
                    k_start = math.ceil((P_MIN - x) / q)
                else:
                    k_start = 0
                if x > P_MAX:
                    continue
                k_end = (P_MAX - x) // q
                for k in range(k_start, k_end + 1):
                    p = x + k * q
                    if P_MIN <= p <= P_MAX:
                        idx = p - P_MIN
                        if i < first_bad[idx]:
                            first_bad[idx] = i

    best_k = 0
    best_p = P_MIN

    for idx in range(size):
        if time.time() > deadline:
            break
        k_candidate = first_bad[idx] - 1
        if k_candidate <= best_k:
            continue
        p = P_MIN + idx
        if not is_prime_arr[p]:
            continue
        valid = True
        for i in range(1, k_candidate + 1):
            n = p * p + i
            rem = n
            for q in primes_p:
                if q > p: break
                if rem % q == 0:
                    while rem % q == 0:
                        rem //= q
            if rem != 1:
                valid = False
                break
        if valid:
            best_k = k_candidate
            best_p = p

    return {'p': best_p, 'k': best_k}
