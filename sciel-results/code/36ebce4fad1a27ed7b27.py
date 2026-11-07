"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 256}
_START_TIME = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _START_TIME

def _pf(n):
    """Prime factorization of n as dict."""
    d = {}
    f = 2
    while f * f <= n:
        while n % f == 0:
            d[f] = d.get(f, 0) + 1
            n //= f
        f += 1
    if n > 1:
        d[n] = d.get(n, 0) + 1
    return d

def _precompute():
    return {n: _pf(n) for n in range(2, P['n'] + 1)}

def _div(a, b, c, pfs):
    """Does a divide b*c?"""
    for p, e in pfs[a].items():
        if e > pfs[b].get(p, 0) + pfs[c].get(p, 0):
            return False
    return True

def _ok(a, b, c, pfs):
    return not (_div(a, b, c, pfs) or _div(b, a, c, pfs) or _div(c, a, b, pfs))

def _can_add(x, S, pfs):
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if not _ok(x, S[i], S[j], pfs):
                return False
    return True

def construct():
    global _START_TIME
    _START_TIME = time.time()
    pfs = _precompute()
    
    primes = [n for n in range(2, P['n'] + 1) if len(pfs[n]) == 1 and list(pfs[n].values())[0] == 1]
    
    S = list(primes)
    comps = [n for n in range(2, P['n'] + 1) if n not in set(primes)]
    comps.sort(key=lambda x: (len(pfs[x]), sum(pfs[x].values()), x))
    
    for c in comps:
        if _elapsed() > _DEADLINE:
            break
        if _can_add(c, S, pfs):
            S.append(c)
    
    best = list(S)
    
    # Local search
    for _ in range(200000):
        if _elapsed() > _DEADLINE:
            break
        improved = False
        for i in range(len(S)):
            saved = S[i]
            S.pop(i)
            for add in range(2, P['n'] + 1):
                if add in S:
                    continue
                if _can_add(add, S, pfs):
                    S.append(add)
                    if len(S) > len(best):
                        best = list(S)
                        improved = True
                    S.pop()
                    break
            S.insert(i, saved)
            if improved:
                break
        if not improved:
            break
    
    if len(best) < len(primes):
        best = primes
    
    # Final check
    S_list = list(best)
    valid = True
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            for k in range(j+1, len(S_list)):
                if not _ok(S_list[i], S_list[j], S_list[k], pfs):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        best = primes
    
    return sorted(best)
# EVOLVE-BLOCK-END
