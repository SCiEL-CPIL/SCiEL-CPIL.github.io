import time
import math

def construct():
    deadline = time.time() + 110

    def is_prime_mr(n):
        if n < 2: return False
        if n == 2 or n == 3: return True
        if n % 2 == 0 or n % 3 == 0: return False
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    best_k = 0
    best_p = 2

    SIEVE_LIMIT = 10**6
    primes_list = sieve_primes(SIEVE_LIMIT)

    for p in primes_list:
        if time.time() > deadline - 5:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            n = base + i
            # Check if n is p-smooth by trial division with primes up to p
            for pr in primes_list:
                if pr > p:
                    break
                while n % pr == 0:
                    n //= pr
                if n == 1:
                    break
            if n == 1 or n <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    return {'p': best_p, 'k': best_k}
