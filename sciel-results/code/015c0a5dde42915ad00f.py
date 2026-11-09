"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def prime_factors(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    factorizations = {n: prime_factors(n) for n in candidates}
    
    def divides_product(a, b, c):
        fa = factorizations[a]
        fb = factorizations[b]
        fc = factorizations[c]
        for p, ea in fa.items():
            eb = fb.get(p, 0)
            ec = fc.get(p, 0)
            if ea > eb + ec:
                return False
        return True
    
    def is_valid_set(S):
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if divides_product(a, b, c):
                        return False
        return True
    
    primes = [n for n in candidates if all(n % d for d in range(2, isqrt(n) + 1))]
    composites = [n for n in candidates if n not in set(primes)]
    
    def can_add(S, x, S_list):
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                if divides_product(x, S_list[i], S_list[j]):
                    return False
        for i in range(n):
            for j in range(i + 1, n):
                if divides_product(S_list[i], x, S_list[j]):
                    return False
                if divides_product(S_list[i], S_list[j], x):
                    return False
        return True
    
    def greedy_add(order):
        S = set(primes)
        S_list = sorted(S)
        for x in order:
            if time.time() > deadline - 3:
                break
            if can_add(S, x, S_list):
                S.add(x)
                S_list.append(x)
                S_list.sort()
        return S
    
    best_set = set(primes)
    
    # Multiple ordering strategies
    S1 = greedy_add(composites)
    if len(S1) > len(best_set):
        best_set = S1
    
    if time.time() < deadline - 10:
        S2 = greedy_add(composites[::-1])
        if len(S2) > len(best_set):
            best_set = S2
    
    if time.time() < deadline - 10:
        def num_pf(n):
            return sum(factorizations[n].values())
        S3 = greedy_add(sorted(composites, key=num_pf))
        if len(S3) > len(best_set):
            best_set = S3
    
    if time.time() < deadline - 10:
        def conflict(n):
            cnt = 0
            for i, p in enumerate(primes):
                for q in primes[i+1:]:
                    if divides_product(n, p, q) or divides_product(p, n, q) or divides_product(p, q, n):
                        cnt += 1
                        break
            return cnt
        S4 = greedy_add(sorted(composites, key=conflict))
        if len(S4) > len(best_set):
            best_set = S4
    
    if time.time() < deadline - 5:
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            random.shuffle(composites)
            Sr = greedy_add(composites)
            if len(Sr) > len(best_set):
                best_set = Sr
    
    result = sorted(best_set)
    
    if not is_valid_set(result):
        result = primes
    
    return result

# EVOLVE-BLOCK-END
