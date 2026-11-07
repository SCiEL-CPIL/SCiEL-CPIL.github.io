import time
import random
from math import gcd

N = 256

def _prime_factors(n):
    d = {}
    x = n
    p = 2
    while p * p <= x:
        while x % p == 0:
            d[p] = d.get(p, 0) + 1
            x //= p
        p += 1
    if x > 1:
        d[x] = d.get(x, 0) + 1
    return d

PF = {n: _prime_factors(n) for n in range(2, N + 1)}

def _covers(a, b, c):
    # does a divide b*c ?
    fa = PF[a]
    fb = PF[b]
    fc = PF[c]
    for p, e in fa.items():
        if fb.get(p, 0) + fc.get(p, 0) < e:
            return False
    return True

def _valid_set(S):
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                    c = lst[k]
                if _covers(a, b, c):
                    return False
    return True

def _can_add(x, S):
    # S is a set; check x can join
    lst = sorted(S)
    n = len(lst)
    # 1) for all distinct b,c in S: x not divide b*c
    for i in range(n):
        b = lst[i]
        for j in range(n):
            if j == i:
                continue
            c = lst[j]
            if _covers(x, b, c):
                return False
    # 2) for all distinct a,c in S: a not divide x*c  (covers a | x*c and a | c*x)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            c = lst[j]
            if _covers(a, x, c):
                return False
    return True

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s kill

    primes = [a for a in range(2, N + 1) if _is_prime(a)]
    best = set(primes)
    best_size = len(best)

    rng = random.Random(12345)

    # Candidate composites sorted in a few orders
    composites = [a for a in range(2, N + 1) if not _is_prime(a)]

    def greedy_add(S, order):
        S = set(S)
        for x in order:
            if x in S:
                continue
            if time.time() > deadline:
                break
            if _can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        # try removing one element, then greedily re-add
        S = set(S)
        cur_size = len(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            lst = sorted(S)
            for rem in lst:
                if time.time() > deadline:
                    break
                S2 = S - {rem}
                # greedily add any element (from all candidates) that fits
                cand = [a for a in range(2, N + 1) if a not in S2]
                rng.shuffle(cand)
                for x in cand:
                    if time.time() > deadline:
                        break
                    if _can_add(x, S2):
                        S2.add(x)
                if len(S2) > cur_size:
                    S = S2
                    cur_size = len(S2)
                    improved = True
                    break
        return S

    # Base greedy with several orderings
    orders = [
        composites,
        sorted(composites, reverse=True),
        sorted(composites, key=lambda x: len(PF[x])),
        sorted(composites, key=lambda x: -len(PF[x])),
        sorted(composites, key=lambda x: x),
        sorted(composites, key=lambda x: -x),
    ]

    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_add(set(primes), list(order))
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Randomized restarts: shuffle composites, greedy, then local search
    while time.time() < deadline - 3:
        order = list(composites)
        rng.shuffle(order)
        S = greedy_add(set(primes), order)
        S = local_search(S)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Final local search on best
    if time.time() < deadline - 2:
        S = local_search(best)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    result = [int(x) for x in sorted(best)]
    return list(result)
