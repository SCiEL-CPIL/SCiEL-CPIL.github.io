"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import sys

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the decomposition reproduces matrix multiplication."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[i][c]) for c in range(rank)] for i in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(p):
                    expected = Frac(1 if j == l else 0)
                    actual = Frac(0)
                    for c in range(rank):
                        actual += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    if actual != expected:
                        return False
    return True

def _try_rank(rank, n, m, p, deadline, rng):
    """Try to find a valid rank-r decomposition. Returns dict or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    iterations = 0
    max_iter = 50000
    while time.time() < deadline - 2 and iterations < max_iter:
        iterations += 1
        # Random small integers for U and V
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(mp)]
        # Solve for W: for each (i,k), solve M W_{k,i} = b
        # M is (m*p) x rank, b is m*p vector with b[j*p+l] = delta_{j,l}
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                # Build system
                rows = []
                rhs = []
                for j in range(m):
                    for l in range(p):
                        row = [Frac(U[i*m+j][c] * V[l*p+k][c]) for c in range(rank)]
                        rows.append(row)
                        rhs.append(Frac(1 if j == l else 0))
                # Gaussian elimination on augmented matrix
                aug = [rows[r] + [rhs[r]] for r in range(m*p)]
                R = m * p
                C = rank
                pivots = []
                col = 0
                for r in range(R):
                    if col >= C:
                        break
                    pr = None
                    for rr in range(r, R):
                        if aug[rr][col] != 0:
                            pr = rr
                            break
                    if pr is None:
                        col += 1
                        continue
                    if pr != r:
                        aug[r], aug[pr] = aug[pr], aug[r]
                    pivots.append((r, col))
                    pv = aug[r][col]
                    if pv != 1:
                        aug[r] = [x / pv for x in aug[r]]
                    for rr in range(R):
                        if rr != r and aug[rr][col] != 0:
                            f = aug[rr][col]
                            aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
                    col += 1
                # Check consistency
                for rr in range(R):
                    if all(aug[rr][c] == 0 for c in range(C)) and aug[rr][C] != 0:
                        ok = False
                        break
                if not ok:
                    break
                # Particular solution
                Wvec = [Frac(0)] * C
                for (r, c) in pivots:
                    Wvec[c] = aug[r][C]
                # Check half-integer constraint
                valid_half = True
                for c in range(C):
                    if Wvec[c].denominator not in (1, 2):
                        valid_half = False
                        break
                if not valid_half:
                    ok = False
                    break
                W[k * n + i] = [int(Wvec[c]) if Wvec[c].denominator == 1 else float(Wvec[c]) for c in range(C)]
        if not ok:
            continue
        # Verify
        if _validate(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # 10s safety margin
    rng = random.Random(42)
    
    # Try ranks from low to high
    # Lower bound is at least max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But realistically much higher. Try from 12 up to 36.
    for rank in range(12, 37):
        if time.time() > deadline - 5:
            break
        result = _try_rank(rank, n, m, p, deadline, rng)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
