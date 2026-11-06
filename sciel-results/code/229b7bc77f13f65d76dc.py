import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 140.0
    
    LIMIT = 10000
    is_prime_small = [True] * (LIMIT + 1)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime_small[i]:
            for j in range(i*i, LIMIT+1, i):
                is_prime_small[j] = False
    small_primes = [i for i, flag in enumerate(is_prime_small) if flag]
    
    def is_prime(n):
        if n < 2: return False
        if n in (2, 3): return True
        if n % 2 == 0 or n % 3 == 0: return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r += m
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def largest_prime_factor(n, p_limit):
        m = n
        max_f = 1
        for q in small_primes:
            if q > p_limit:
                break
            if m % q == 0:
                max_f = max(max_f, q)
                while m % q == 0:
                    m //= q
        if m == 1:
            return max_f
        if is_prime(m):
            return max(max_f, m)
        
        def _lpf(x):
            if x == 1:
                return 1
            if is_prime(x):
                return x
            d = pollard_brent(x)
            return max(_lpf(d), _lpf(x // d))
            
        return max(max_f, _lpf(m))

    best_k = 0
    best_p = 0
    
    CAND_LIMIT = 2 * 10**6
    is_prime_cand = [True] * (CAND_LIMIT + 1)
    is_prime_cand[0] = is_prime_cand[1] = False
    for i in range(2, int(CAND_LIMIT**0.5) + 1):
        if is_prime_cand[i]:
            for j in range(i*i, CAND_LIMIT+1, i):
                is_prime_cand[j] = False
    candidates = [i for i, flag in enumerate(is_prime_cand) if flag]
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if is_prime(n):
                if n > p:
                    break
                else:
                    k += 1
                    continue
            
            lpf = largest_prime_factor(n, p)
            if lpf > p:
                break
            else:
                k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
