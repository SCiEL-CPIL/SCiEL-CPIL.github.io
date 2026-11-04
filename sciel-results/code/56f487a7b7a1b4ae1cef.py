"""Search for a rank-minimal exact tensor decomposition of 3x3 * 3x5 matrix mult."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _mul_tensor(n, m, p):
    # T[k*n+i, j*m+? ...] -> index (a=i, b=j, c=k); value 1 if a,b,c consistent.
    # We'll build dict of (i,j,k) -> 1 for all triples.
    return {(i, j, k): 1 for i in range(n) for j in range(m) for k in range(p)}

def _verify(U, V, W, n, m, p, rank):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Build sum dict
    sums = {}
    for r in range(rank):
        uc = U[r]
        vc = V[r]
        wc = W[r]
        # U rows: i*m+j ; V rows: j*p+k ; W rows: k*n+i
        # iterate over nonzero entries
        for ui in range(n * m):
            uval = uc[ui]
            if uval == 0:
                continue
            i = ui // m
            j = ui % m
            for vk in range(m * p):
                vval = vc[vk]
                if vval == 0:
                    continue
                if vk // p != j:
                    continue
                k = vk % p
                widx = k * n + i
                wval = wc[widx]
                if wval == 0:
                    continue
                key = (i, j, k)
                sums[key] = sums.get(key, 0) + uval * vval * wval
    # Every (i,j,k) must equal 1; no extra keys.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if sums.get((i, j, k), 0) != 1:
                    return False
    if len(sums) != n * m * p:
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

# Candidate search over small-rank structured decompositions using half-integers.
def _search(n, m, p, deadline):
    best = None
    rng = random.Random(12345)
    # We'll attempt random rank-r decompositions with entries in {0, ±1/2, ±1}
    # and verify. Try increasing rank from a low bound.
    low = n * m * p - 1  # start just below schoolbook
    # Actually try a range of ranks; but verification is cheap. Search ranks.
    # Use a smarter lower bound: rank >= max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    # Try ranks from lb upward but we want minimal; do a few random trials per rank.
    for rank in range(lb, n * m * p):
        if time.time() > deadline:
            break
        trials = 2000
        for t in range(trials):
            if time.time() > deadline:
                break
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for r in range(rank):
                for i in range(n * m):
                    U[i][r] = rng.choice([0, 0, 0, 1, -1, 0.5, -0.5])
                for j in range(m * p):
                    V[j][r] = rng.choice([0, 0, 0, 1, -1, 0.5, -0.5])
                for k in range(p * n):
                    W[k][r] = rng.choice([0, 0, 0, 1, -1, 0.5, -0.5])
            if _verify(U, V, W, n, m, p, rank):
                best = {"U": U, "V": V, "W": W, "rank": rank}
                return best
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin under 120s
    # Try a structured search; if it finds something, use it.
    found = _search(n, m, p, deadline)
    if found is not None:
        return {"U": found["U"], "V": found["V"], "W": found["W"]}
    # Fallback: schoolbook (always valid).
    return _schoolbook(n, m, p)
