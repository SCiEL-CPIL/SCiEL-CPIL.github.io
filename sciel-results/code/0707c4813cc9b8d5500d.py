import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Pre-compute primes up to 10^5 for fast division
    LIMIT = 10**5
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # Miller-Rabin for large numbers
    def is_prime_mr(n):
        if n < 2: return False
        # Check small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p: return True
            if n % p == 0: return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        # Divide by small primes
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            if is_prime_mr(temp):
                result -= result // temp
            else:
                # Fallback: trial division for composite remainder
                # This is rare
                p = primes[-1] + 2
                while p * p <= temp:
                    if temp % p == 0:
                        result -= result // p
                        while temp % p == 0:
                            temp //= p
                    p += 2
                if temp > 1:
                    result -= result // temp
        return result

    def check_window(s, max_len=200, max_n=10**12):
        seen = set()
        k = 0
        for i in range(max_len):
            v = s + i
            if v > max_n:
                break
            pv = phi(v)
            if pv in seen:
                break
            seen.add(pv)
            k += 1
        return k

    best_k = 1
    best_s = 1
    
    max_n = 10**12

    # Strategy 1: Check small starts
    for s in range(1, 500):
        if time.time() - start_time > DEADLINE:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 2: Random search
    # Reduced iterations to ensure time budget
    random.seed(42)
    for _ in range(1000):
        if time.time() - start_time > DEADLINE - 5:
            break
        s = random.randint(1, max_n - 200)
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    return {'start': best_s, 'length': best_k}
