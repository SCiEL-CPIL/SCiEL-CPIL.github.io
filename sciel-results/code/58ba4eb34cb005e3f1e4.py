"""Rewritten program: search-based maximization of multiplicatively independent
integers in 2..40 (distinct subset products <=> rational linear independence of
prime-exponent vectors)."""
# EVOLVE-BLOCK-START
import time

# ---- fixed problem data (computed once at import) ----
def _sieve_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

_PRIMES = _sieve_primes(40)
_PLEN = len(_PRIMES)

def _exp_vec(n):
    v = [0] * _PLEN
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    return tuple(v)

_CAND = list(range(2, 41))
_VECS = [_exp_vec(n) for n in _CAND]
_VLEN = len(_CAND)

# rank of a list of exponent vectors (rational linear independence)
def _rank(vecs):
    M = [list(v) for v in vecs]
    r = 0
    for c in range(_PLEN):
        piv = -1
        for i in range(r, len(M)):
            if M[i][c]:
                piv = i
                break
        if piv < 0:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c] / pv
                for j in range(c, _PLEN):
                    M[i][j] -= f * M[r][j]
        r += 1
        if r == len(M):
            break
    return r

# ---- greedy independent set (deterministic, fast) ----
def _greedy():
    chosen = []
    rank = 0
    for i in range(_VLEN):
        if rank >= _PLEN:
            break
        if _rank(chosen + [_VECS[i]]) > rank:
            chosen.append(_VECS[i])
            rank += 1
    return chosen

# ---- bounded branch-and-bound over the candidate index set ----
def _branch_and_bound(start, chosen, best):
    rank = _rank(chosen)
    if len(chosen) > len(best):
        best.append(list(chosen))
    if rank >= _PLEN:
        return
    # upper bound: chosen + (independent room) + (remaining count)
    if len(chosen) + (_PLEN - rank) + (_VLEN - start) <= len(best):
        return
    for i in range(start, _VLEN):
        if _rank(chosen + [_VECS[i]]) > rank:
            _branch_and_bound(i + 1, chosen + [_VECS[i]], best)

def _search_best(deadline):
    best = _greedy()
    t0 = time.monotonic()
    def over():
        return (time.monotonic() - t0) > deadline
    # search for a set strictly larger than the greedy one
    for start in range(_VLEN):
        if over():
            break
        _branch_and_bound(start, [_VECS[start]], best)
        if len(best) > _PLEN:
            break
    return best

# ---- public entry point ----
def construct():
    deadline = 70.0  # safety margin below the 90s kill
    best = _search_best(deadline)
    # convert back to integers (the candidate whose vector was chosen)
    vmap = {v: n for n, v in zip(_CAND, _VECS)}
    result = [vmap[v] for v in best]
    # sanity: guarantee a valid, distinct, in-range list of Python ints
    seen = set()
    clean = []
    for x in result:
        x = int(x)
        if 2 <= x <= 40 and x not in seen:
            seen.add(x)
            clean.append(x)
    if not clean:
        clean = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return list(clean)
# EVOLVE-BLOCK-END
