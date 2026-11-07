import time
from math import isqrt
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    n = 256
    
    # Sieve primes
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n+1) if is_prime[i]]
    composites = [x for x in range(2, n+1) if not is_prime[x]]
    
    # Prime factorization: pf[x] = list of (p, exponent)
    pf = [[] for _ in range(n + 1)]
    for p in primes:
        for j in range(p, n+1, p):
            exp = 0
            tmp = j
            while tmp % p == 0:
                exp += 1
                tmp //= p
            if exp > 0:
                pf[j].append((p, exp))
    
    # v_p[x] = exponent of prime p in x
    # Build a dict: for each prime p, v_p[x] for all x
    prime_list = primes
    v_exp = {}  # v_exp[p] = list of exponents for x=0..n
    for p in prime_list:
        v_exp[p] = [0] * (n + 1)
        for j in range(p, n+1, p):
            tmp = j
            e = 0
            while tmp % p == 0:
                e += 1
                tmp //= p
            v_exp[p][j] = e
    
    def valid_set(S):
        """Check validity of set S."""
        lst = list(S)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(m):
                if j == i: continue
                b = lst[j]
                for k in range(m):
                    if k == i or k == j: continue
                    c = lst[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    def fast_add_check(S_list, x, S_set):
        """Check if x can be added to S. S_list is list of S, S_set is set of S."""
        m = len(S_list)
        # x as 'a': check x | b*c for all distinct b,c in S
        for i in range(m):
            b = S_list[i]
            for j in range(i+1, m):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
        # x as 'b' or 'c': check a | x*d for all a in S, d in S, a!=d, a!=x
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i: continue
                d = S_list[j]
                if (x * d) % a == 0:
                    return False
        return True
    
    # Baseline: all primes
    best_set = set(primes)
    best_size = len(primes)
    
    # Try adding composites greedily
    S_list = list(best_set)
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for x in composites:
            if x in best_set: continue
            if fast_add_check(S_list, x, best_set):
                best_set.add(x)
                S_list.append(x)
                improved = True
    
    # Local search: try swaps
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            S_list = list(best_set)
            m = len(S_list)
            for i in range(m):
                if time.time() > deadline - 15: break
                elem = S_list[i]
                trial = set(best_set)
                trial.remove(elem)
                trial_list = list(trial)
                for x in composites:
                    if x in trial: continue
                    if fast_add_check(trial_list, x, trial):
                        trial.add(x)
                        if len(trial) > len(best_set):
                            best_set = trial
                            improved = True
                        break
    
    # Final verification
    if not valid_set(best_set):
        best_set = set(primes)
    
    result = [int(x) for x in sorted(best_set)]
    return result
