import time
import math

def construct():
    deadline = time.time() + 115

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

    SIEVE_LIMIT = 10**7
    primes_list = sieve_primes(SIEVE_LIMIT)
    
    # Helper to check if n is p-smooth
    # Returns True if all prime factors of n are <= p
    def is_p_smooth(n, p, primes):
        if n == 1:
            return True
        for q in primes:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
            if n == 1:
                return True
        return n <= p

    def check_run_smooth(base, p, max_k=200):
        vals = [base + i for i in range(1, max_k + 1)]
        
        for pr in primes_list:
            if pr > p:
                break
            rem = (-base) % pr
            if rem == 0:
                start = pr
            else:
                start = rem
            if start > max_k:
                continue
            idx = start - 1
            while idx < max_k:
                v = vals[idx]
                while v % pr == 0:
                    v //= pr
                vals[idx] = v
                idx += pr
        
        k = 0
        for i in range(max_k):
            v = vals[i]
            if v == 1:
                k += 1
            else:
                break
        return k

    # Phase 1: Quick scan
    for p in primes_list:
        if time.time() > deadline - 10:
            break
        
        # Quick check: is p^2+1 p-smooth?
        if not is_p_smooth(p*p + 1, p, primes_list):
            continue
            
        base = p * p
        k = check_run_smooth(base, p)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    return {'p': best_p, 'k': best_k}
