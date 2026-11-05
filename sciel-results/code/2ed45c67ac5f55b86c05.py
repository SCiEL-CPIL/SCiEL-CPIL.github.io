import time
from math import factorial

_START = time.time()
_TIMEOUT = 80.0

def _time_left():
    return _TIMEOUT - (time.time() - _START)

TARGET = factorial(12)
PRIMES = (2, 3, 5, 7, 11)

def _divisors(n):
    divs = [1]
    for p in PRIMES:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        if e == 0:
            continue
        mult = [p ** k for k in range(e + 1)]
        new = []
        for d in divs:
            for m in mult:
                new.append(d * m)
        divs = new
    return divs

_ALL_DIVS = _divisors(TARGET)

def _solve(M):
    cands = [d for d in _ALL_DIVS if 12 < d <= M]
    if not cands:
        return None
    cands.sort()
    used = set()
    path = []
    dead = set()
    n = len(cands)

    def dfs(rem, min_idx):
        if _time_left() < 1.0:
            return None
        if rem == 1:
            return list(path)
        if 1 < rem <= 12:
            return None
        if len(path) >= 12:
            return None
        if min_idx >= n:
            return None
        if rem < cands[min_idx]:
            return None
        key = (rem, min_idx)
        if key in dead:
            return None
        # if rem itself is a valid unused factor
        if rem <= M and rem not in used and rem > 12:
            return path + [rem]
        for i in range(min_idx, n):
            d = cands[i]
            if d > rem:
                break
            if d in used:
                continue
            if rem % d != 0:
                continue
            used.add(d)
            path.append(d)
            res = dfs(rem // d, i + 1)
            path.pop()
            used.remove(d)
            if res is not None:
                return res
        dead.add(key)
        return None

    return dfs(TARGET, 0)

def construct():
    # find minimal M
    for M in range(13, 1000000):
        if _time_left() < 3.0:
            break
        sol = _solve(M)
        if sol is not None:
            return sol
    # fallback: single factor
    return [TARGET]
