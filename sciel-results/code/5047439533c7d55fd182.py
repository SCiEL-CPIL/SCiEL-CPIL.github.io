"""Structural rewrite: randomized algebraic rank reduction for the 2x5*5x6 tensor."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
_DEADLINE = 150.0
_START = None

def _elapsed():
    return time.monotonic() - _START

def _mulmat(A, B):
    n = len(A); m = len(A[0]); p = len(B[0])
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(m):
            aij = Ai[j]
            if aij:
                Bj = B[j]
                for k in range(p):
                    Ci[k] += aij * Bj[k]
    return C

def _verify(U, V, W, n, m, p, r):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(r):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if s != 1:
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
    return U, V, W, rank

# ---------- randomized algebraic rank reduction ----------
def _rand_frac(rng, sparsity, scale):
    if rng.random() < sparsity:
        v = rng.randint(-scale, scale)
        return Fraction(v, 1)
    return Fraction(0, 1)

def _bernstein_try(r, n, m, p, rng, sparsity, scale, max_iter):
    # Build random sparse U (nm x r), V (mp x r). Then solve for W.
    nm = n * m
    mp = m * p
    pn = p * n
    for it in range(max_iter):
        U = [[_rand_frac(rng, sparsity, scale) for _ in range(r)] for _ in range(nm)]
        V = [[_rand_frac(rng, sparsity, scale) for _ in range(r)] for _ in range(mp)]
        # Build linear system for W: for each (i,k), sum_j U[i*m+j]*V[j*p+k]*W[k*n+i] = 1
        # Unknowns are W rows indexed by (k*n+i), total pn. Each equation couples them.
        # Equation for (i,k): sum_j (U[i*m+j]*V[j*p+k]) * W[k*n+i] = 1
        # But W[k*n+i] is a single unknown per (k,i). So each equation is independent!
        # sum_j U[i*m+j]*V[j*p+k] * W[k*n+i] = 1  =>  W[k*n+i] = 1 / sum_j(...)
        # This only works because each (i,k) involves exactly one W unknown.
        W = [[Fraction(0,1) for _ in range(r)] for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                denom = Fraction(0,1)
                for j in range(m):
                    denom += U[i*m+j][0] * 0  # placeholder
                # recompute properly
                denom = Fraction(0,1)
                for j in range(m):
                    denom += U[i*m+j][0] * V[j*p+k][0]
                # NOTE: this uses only column 0; need full r-dim dot for general r.
                # For r=1 the above is correct. For r>1 the system is coupled.
                if denom == 0:
                    ok = False
                    break
                W[k*n+i][0] = Fraction(1,1) / denom
            if not ok:
                break
        if not ok:
            continue
        # verify
        if _verify(U, V, W, n, m, p, r):
            return U, V, W, r
    return None

# General r: solve the coupled linear system via Gaussian elimination over Fractions.
def _solve_linear(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = None
        for r2 in range(c, n):
            if M[r2][c] != 0:
                piv = r2
                break
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        inv = Fraction(1,1) / M[c][c]
        for j in range(c, n + 1):
            M[c][j] *= inv
        for r2 in range(n):
            if r2 != c and M[r2][c] != 0:
                f = M[r2][c]
                for j in range(c, n + 1):
                    M[r2][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]

def _bernstein_general(r, n, m, p, rng, sparsity, scale, max_iter):
    nm = n * m
    mp = m * p
    pn = p * n
    for it in range(max_iter):
        U = [[_rand_frac(rng, sparsity, scale) for _ in range(r)] for _ in range(nm)]
        V = [[_rand_frac(rng, sparsity, scale) for _ in range(r)] for _ in range(mp)]
        # Unknowns: W[k*n+i][t], t in range(r). Total pn*r unknowns.
        # Equations: for each (i,k,t): sum_j U[i*m+j][t]*V[j*p+k][t]*W[k*n+i][t] = 1
        # This decouples by t! For each t, W_t is determined independently:
        # sum_j U_t[i*m+j]*V_t[j*p+k]*W_t[k*n+i] = 1  =>  W_t[k*n+i] = 1/(sum_j ...)
        W = [[Fraction(0,1) for _ in range(r)] for _ in range(pn)]
        ok = True
        for t in range(r):
            for i in range(n):
                for k in range(p):
                    denom = Fraction(0,1)
                    for j in range(m):
                        denom += U[i*m+j][t] * V[j*p+k][t]
                    if denom == 0:
                        ok = False
                        break
                    W[k*n+i][t] = Fraction(1,1) / denom
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        if _verify(U, V, W, n, m, p, r):
            return U, V, W, r
    return None

# ---------- lower-bound certificate search ----------
def _lower_bound_cert(n, m, p, rng, tries):
    # Find M (dense random) with sum M*T = 0 but M is not orthogonal to every rank-1.
    # If such M exists, any decomposition must use a term not in ker(M) -> rank > 0 trivially.
    # To bound rank >= r, need M orthogonal to all but (r-1) rank-1s -> hard.
    # We use a weaker but useful check: if tensor is NOT a sum of (r-1) rank-1s for some r,
    # we cannot easily prove it. Instead we just try to find a decomposition of rank < schoolbook.
    return None

# ---------- main ----------
def construct():
    global _START
    _START = time.monotonic()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)

    # Start from schoolbook as the safe fallback.
    U0, V0, W0, rank0 = _schoolbook(n, m, p)
    best = (rank0, U0, V0, W0)

    # Try to beat schoolbook via randomized algebraic search, descending rank.
    # A known lower bound for matrix-mult tensor rank is max(n*m, m*p, p*n) when one dim=1;
    # here none is 1, so we search a modest window below schoolbook.
    max_rank = rank0
    # We will attempt ranks from a small set below schoolbook.
    candidates = []
    # Always include a few ranks well below schoolbook.
    for r in range(1, max_rank + 1):
        if r in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 45, 48, 50, 55, 60):
            candidates.append(r)
    candidates = sorted(set(candidates))

    # Time budget: spend most of it on the most promising (lowest) ranks first.
    for r in candidates:
        if r >= best[0]:
            break
        if _elapsed() > _DEADLINE - 5.0:
            break
        # Try a handful of randomizations for this rank.
        sparsities = [0.5, 0.7, 0.3]
        scales = [2, 3, 5]
        found = None
        for sp in sparsities:
            if _elapsed() > _DEADLINE - 5.0:
                break
            for sc in scales:
                if _elapsed() > _DEADLINE - 5.0:
                    break
                max_iter = 40
                res = _bernstein_general(r, n, m, p, rng, sp, sc, max_iter)
                if res is not None:
                    found = res
                    break
            if found:
                break
        if found:
            U, V, W, rr = found
            if _verify(U, V, W, n, m, p, rr):
                best = (rr, U, V, W)

    # Deadline guard: return the best verified solution.
    if _elapsed() > _DEADLINE - 2.0:
        r, U, V, W = best
        return {"U": U, "V": V, "W": W}

    return {"U": best[1], "V": best[2], "W": best[3]}
