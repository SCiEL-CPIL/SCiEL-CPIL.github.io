"""Search for n <= 1e18 maximizing tau_perp(n) = # adjacent coprime divisor pairs."""
import math
import time

DEADLINE = 140.0  # safety margin under 150s
_START = time.time()

def _remaining():
    return DEADLINE - (time.time() - _START)

def _divisors_from_exp(primes, exps):
    """Generate all divisors from prime/exponent lists."""
    divs = [1]
    for p, e in zip(primes, exps):
        cur = []
        pw = 1
        for _ in range(e + 1):
            for d in divs:
                cur.append(d * pw)
            pw *= p
        divs = cur
    return divs

def _tau_perp(n, primes, exps):
    divs = _divisors_from_exp(primes, exps)
    divs.sort()
    cnt = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            cnt += 1
    return cnt

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    LIMIT = 10**18

    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = 1  # seed score baseline (tau_perp of seed)
    # Actually compute seed's real tau_perp for consistency of returned n; we return n, evaluator scores it.
    # We just need to return a good n.

    def dfs(idx, product, exps):
        nonlocal best_n, best_score
        if _remaining() < 2.0:
            return
        if idx == len(primes):
            # evaluate
            if product < 2:
                return
            # limit divisor count to keep eval fast
            tau = 1
            for e in exps:
                tau *= (e + 1)
            if tau > 200000:
                return
            s = _tau_perp(product, primes[:idx], exps)
            if s > best_score:
                best_score = s
                best_n = product
            return
        p = primes[idx]
        # try exponents 0..max such that product * p^e <= LIMIT
        val = product
        e = 0
        while True:
            # prune: even with all remaining primes at exp 1, can we still reach >=2? not needed
            dfs(idx + 1, val, exps + [e])
            if val > LIMIT // p:
                break
            val *= p
            e += 1
            if e > 60:
                break

    # To keep search tractable, restrict to first 8 primes with bounded exponents,
    # plus explore squarefree subsets of more primes.
    # Phase 1: squarefree over first 12 primes (2^12 = 4096 candidates)
    for mask in range(1, 1 << 12):
        if _remaining() < 2.0:
            break
        product = 1
        exps = []
        for i in range(12):
            if mask & (1 << i):
                product *= primes[i]
                exps.append(1)
            else:
                exps.append(0)
        if product > LIMIT or product < 2:
            continue
        s = _tau_perp(product, primes[:12], exps)
        if s > best_score:
            best_score = s
            best_n = product

    # Phase 2: bounded-exponent DFS over first 6 primes
    if _remaining() > 3.0:
        dfs(0, 1, [])

    return best_n

# EVOLVE-BLOCK-END
