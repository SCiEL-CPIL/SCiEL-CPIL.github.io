import time
import random
import math
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _mulberry32(seed):
    state = seed & 0xFFFFFFFF
    def gen():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = (t ^ (t >> 15)) * (t | 1) & 0xFFFFFFFF
        t = (t ^ (t >> 13)) * (t | 0x5BD1E995) & 0xFFFFFFFF
        t = t ^ (t >> 16)
        return (t & 0x7FFFFFFF) / 0x80000000
    return gen

def _build_patterns(n, m, p):
    """Enumerate candidate rank-1 patterns (S_A, S_B, S_C)."""
    nm = n * m
    mp = m * p
    pn = p * n
    patterns = []
    for sa in range(1, 1 << nm):
        for sb in range(1, 1 << mp):
            for sc in range(1, 1 << pn):
                patterns.append((sa, sb, sc))
    return patterns

def _coeffs(sa, sb, sc, n, m, p):
    """Coefficient of pattern (sa,sb,sc) for each (i,j,k)."""
    nm = n * m
    mp = m * p
    pn = p * n
    out = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = 1 if (sa >> (i * m + j)) & 1 else 0
                b = 1 if (sb >> (j * p + k)) & 1 else 0
                c = 1 if (sc >> (k * n + i)) & 1 else 1 - 1  # placeholder
                c = 1 if (sc >> (k * n + i)) & 1 else 0
                out.append(a * b * c)
    return out

def _dot(u, v):
    s = 0
    for i in range(len(u)):
        s += u[i] * v[i]
    return s

def _norm(u):
    s = 0
    for x in u:
        s += x * x
    return math.sqrt(s)

def _lll_reduce(M, delta=0.99, eta=0.51):
    """LLL on a list of integer row vectors. Returns reduced basis (list of rows)."""
    # M: list of rows (lists of ints)
    rows = [list(r) for r in M]
    # Remove zero rows
    rows = [r for r in rows if any(x != 0 for x in r)]
    if not rows:
        return []
    # Sort by norm
    rows.sort(key=lambda r: _norm(r))
    B = [rows[0]]
    mu = [0.0]  # mu[i] = <B[i], B[0>] / <B[0],B[0>]
    beta0 = _norm(B[0]) ** 2
    for i in range(1, len(rows)):
        v = list(rows[i])
        k = len(B) - 1
        # Insert v into basis
        while k >= 0:
            bk = B[k]
            beta_k = _norm(bk) ** 2
            if beta_k == 0:
                k -= 1
                continue
            mu_k = _dot(v, bk) / beta_k
            v = [v[j] - int(round(mu_k)) * bk[j] for j in range(len(bk))]
            if _norm(v) < eta * _norm(bk):
                # v is short relative to bk; shift bk down
                B[k + 1] = bk
                B[k] = v
                # recompute mu for new position
                if k > 0:
                    bk2 = B[k - 1]
                    beta_k2 = _norm(bk2) ** 2
                    if beta_k2 > 0:
                        mu.insert(k, _dot(B[k], bk2) / beta_k2)
                    else:
                        mu.insert(k, 0.0)
                else:
                    mu.insert(0, 0.0)
                # size reduction for new position
                if k > 0:
                    bk2 = B[k - 1]
                    beta_k2 = _norm(bk2) ** 2
                    if beta_k2 > 0:
                        mu_k = _dot(B[k], bk2) / beta_k2
                        B[k] = [B[k][j] - int(round(mu_k)) * bk2[j] for j in range(len(bk2))]
                k -= 1
                continue
            else:
                # size reduce against B[k]
                if beta_k > 0:
                    mu_k = _dot(v, bk) / beta_k
                    v = [v[j] - int(round(mu_k)) * bk[j] for j in range(len(bk))]
                # now v is orthogonal-ish; append
                B.append(v)
                mu.append(0.0)
                break
        # Lovasz condition check (simplified)
        for idx in range(len(B) - 1, 0, -1):
            if idx >= len(B):
                break
            b_prev = B[idx - 1]
            b_cur = B[idx]
            beta_prev = _norm(b_prev) ** 2
            if beta_prev == 0:
                continue
            mu_val = _dot(b_cur, b_prev) / beta_prev
            # size reduce
            b_cur = [b_cur[j] - int(round(mu_val)) * b_prev[j] for j in range(len(b_prev))]
            B[idx] = b_cur
            # Lovasz
            if _norm(b_cur) ** 2 >= (delta - mu_val * mu_val) * beta_prev:
                break
            else:
                # swap
                B[idx - 1], B[idx] = B[idx], B[idx - 1]
    return B

def _find_solution(A, patterns, budget, rng):
    """Search for a sparse integer combination of patterns that equals the all-ones target."""
    # A: list of coefficient vectors (one per pattern), each length N
    N = len(A[0]) if A else 0
    target = [1] * N
    # Try random subsets of patterns, solve integer least squares via rounding
    # For a subset S, build matrix M (N x |S|), solve M x = target over reals, round to int, verify
    # Use small subsets first (rank = |S|)
    best_rank = None
    best_sol = None
    num_patterns = len(A)
    # Greedy: try to find a solution with small rank
    # For rank r, pick r patterns and check if there's an integer solution
    # We'll use a randomized approach: pick random subsets, solve via Gaussian elimination with fractions
    max_rank_to_try = min(40, num_patterns)
    for r in range(1, max_rank_to_try + 1):
        if budget.elapsed() > budget.deadline - 5:
            break
        # Try several random subsets of size r
        attempts = 50
        for _ in range(attempts):
            if budget.elapsed() > budget.deadline - 3:
                break
            idxs = rng.sample(range(num_patterns), r) if r < num_patterns else list(range(num_patterns))
            # Build system: for each (i,j,k): sum_{t in idxs} A[pattern_t][ijk] * x_t = 1
            # M is N x r
            M = [A[idxs[t]] for t in range(r)]  # list of r vectors? No: M[row][col]
            # M[row][col] = A[col][row]
            # Solve M^T x = target  (r x N system, N equations, r unknowns)
            # Use least squares / exact: we want integer x such that M^T x = target
            # Form N x r matrix B where B[row][col] = A[col][row]
            Bmat = [[A[col][row] for col in range(r)] for row in range(N)]
            # Solve Bmat x = target over rationals (if consistent)
            # Use Gaussian elimination with fractions on the augmented matrix
            # But N can be large (100) and r small; overdetermined. Check consistency.
            # Do Gaussian elimination on Bmat augmented with target
            aug = [Bmat[row] + [1] for row in range(N)]
            # Eliminate
            rank_found = 0
            for col in range(r):
                # Find pivot
                pivot = None
                for row in range(rank_found, N):
                    if aug[row][col] != 0:
                        pivot = row
                        break
                if pivot is None:
                    continue
                aug[rank_found], aug[pivot] = aug[pivot], aug[rank_found]
                pv = aug[rank_found][col]
                # Normalize
                for c in range(col, r + 1):
                    aug[rank_found][c] = aug[rank_found][c] / pv
                for row in range(N):
                    if row != rank_found and aug[row][col] != 0:
                        f = aug[row][col]
                        for c in range(col, r + 1):
                            aug[row][c] = aug[row][c] - f * aug[rank_found][c]
                rank_found += 1
            # Check consistency: any row with all zeros in cols but nonzero in target?
            consistent = True
            for row in range(N):
                if all(aug[row][c] == 0 for c in range(r)) and aug[row][r] != 0:
                    consistent = False
                    break
            if not consistent:
                continue
            # Solution: x[col] = aug[col][r] for col < rank_found, free vars = 0
            x = [0] * r
            for col in range(rank_found):
                x[col] = aug[col][r]
            # Check if integer
            if all(abs(x[c] - round(x[c])) < 1e-9 for c in range(r)):
                x = [int(round(x[c])) for c in range(r)]
                # Verify exactly
                ok = True
                for row in range(N):
                    s = 0
                    for col in range(r):
                        s += A[col][row] * x[col]
                    if s != 1:
                        ok = False
                        break
                if ok:
                    best_rank = r
                    best_sol = (idxs, x)
                    return best_rank, best_sol
        # If we found a solution at this rank, return
        if best_sol is not None:
            return best_rank, best_sol
    return best_rank, best_sol

def _schoolbook(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                if abs(s - 1.0) > tol:
                    return False
    return True

class _Budget:
    def __init__(self, limit):
        self.start = time.time()
        self.deadline = self.start + limit
    def elapsed(self):
        return time.time() - self.start

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    N = n * m * p  # 100

    budget = _Budget(140)  # safety margin before 150s

    # Start with schoolbook
    U, V, W, school_rank = _schoolbook(n, m, p)

    # Try to find a better solution via pattern enumeration + linear algebra
    # Enumerate patterns
    patterns = _build_patterns(n, m, p)
    rng = _mulberry32(12345)

    # Build coefficient matrix: A[pattern][ijk]
    A = []
    for (sa, sb, sc) in patterns:
        A.append(_coeffs(sa, sb, sc, n, m, p))

    # Try to find a sparse solution
    best_rank, best_sol = _find_solution(A, patterns, budget, rng)

    if best_sol is not None:
        idxs, x = best_sol
        r = len(x)
        U2 = [[0] * r for _ in range(nm)]
        V2 = [[0] * r for _ in range(mp)]
        W2 = [[0] * r for _ in range(pn)]
        for t in range(r):
            if x[t] == 0:
                continue
            sa, sb, sc = patterns[idxs[t]]
            coeff = x[t]
            for i in range(n):
                for j in range(m):
                    if (sa >> (i * m + j)) & 1:
                        U2[i * m + j][t] = coeff
            for j in range(m):
                for k in range(p):
                    if (sb >> (j * p + k)) & 1:
                        V2[j * p + k][t] = 1
            for k in range(p):
                for i in range(n):
                    if (sc >> (k * n + i)) & 1:
                        W2[k * n + i][t] = 1
        # Verify
        if _verify(U2, V2, W2, n, m, p):
            return {"U": U2, "V": V2, "W": W2}

    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
