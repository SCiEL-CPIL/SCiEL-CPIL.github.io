import math
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _validate(lst, n=210, max_window=10210, max_size=100):
    if not isinstance(lst, (list, tuple)):
        return False
    if len(lst) < 1 or len(lst) > max_size:
        return False
    if len(set(lst)) != len(lst):
        return False
    for x in lst:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= n or x > max_window:
            return False
    prod = n
    for x in lst:
        prod *= x
    return _is_square(prod)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    n = 210
    max_window = 10210
    
    def sieve_primes(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit+1, i):
                    is_p[j] = False
        return [i for i in range(2, limit+1) if is_p[i]]
    
    primes = sieve_primes(max_window)
    
    def parity_set(num):
        k = num
        ps = set()
        for p in primes:
            if p * p > k:
                break
            if k % p == 0:
                exp = 0
                while k % p == 0:
                    k //= p
                    exp += 1
                if exp % 2 == 1:
                    ps.add(p)
        if k > 1:
            ps.add(k)
        return frozenset(ps)
    
    target_ps = parity_set(n)
    
    ps_cache = {}
    for v in range(n+1, max_window+1):
        ps_cache[v] = parity_set(v)
    
    best_result = None
    best_val = None
    
    for M in range(n+2, 400):
        if time.time() > deadline - 5:
            break
        
        needed = frozenset(target_ps.symmetric_difference(ps_cache[M]))
        
        achievable = {frozenset(): ()}
        
        for v in range(n+1, M):
            if time.time() > deadline - 5:
                break
            pv = ps_cache[v]
            updates = {}
            for xv, subset in achievable.items():
                new_xv = frozenset(xv.symmetric_difference(pv))
                new_subset = subset + (v,)
                if new_xv not in updates:
                    updates[new_xv] = new_subset
            for xv, subset in updates.items():
                if xv not in achievable:
                    achievable[xv] = subset
            
            if needed in achievable:
                subset = achievable[needed]
                if len(subset) > 0:
                    lst = list(subset) + [M]
                    if len(lst) <= 100 and _validate(lst):
                        best_result = lst
                        best_val = M - n
                        break
        
        if best_result is not None:
            break
    
    if best_result is None:
        for t in range(1, 300):
            if time.time() > deadline - 5:
                break
            prod = 210 * t * t
            best_pair = None
            a = 1
            while a * a <= prod:
                if prod % a == 0:
                    b = prod // a
                    if a > 210 and b > 210 and a != b:
                        mx = max(a, b)
                        if best_pair is None or mx < best_pair[0]:
                            best_pair = (mx, [a, b])
                a += 1
            if best_pair is not None:
                best_result = best_pair[1]
                best_val = best_pair[0] - n
                break
    
    if best_result is None:
        best_result = [840]
        best_val = 630
    
    if not _validate(best_result):
        best_result = [840]
    
    return [int(x) for x in best_result]
