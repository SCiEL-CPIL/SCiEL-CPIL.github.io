"""Fresh design: structured rank-reduction search for 2x5x5 matrix multiplication.

Strategy:
  1. Start from the schoolbook rank-50 decomposition.
  2. Use exact rational arithmetic (Fraction) to verify exactness of the
     multiplication tensor: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1.
  3. Try to reduce rank by:
       a) Removing redundant columns (linear dependence in the tensor).
       b) Strassen-style linear-combination packing for the n=2 dimension.
       c) Randomized low-rank factor search with exact verification.
  4. Return the lowest-rank exact decomposition found, with a deadline guard.
"""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 5, 5]

_DEADLINE = time.time() + 120.0
_MARGIN = 10.0  # seconds of safety margin


def _time_left():
    return _DEADLINE - _MARGIN - time.time()


def _tensor_entry(i, j, k, n, m, p):
    """Canonical multiplication tensor entry: 1 for all (i,j,k)."""
    return 1


def _verify_exact(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k.
    U,V,W are lists of lists of Fraction (or int/float converted)."""
    rank = len(U[0])
    # Convert to Fraction for exactness if not already
    def to_frac(x):
        if isinstance(x, Fraction):
            return x
        return Fraction(x).limit_denominator(10**6)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]

    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c = k * n + i
                s = Fraction(0)
                for r in range(rank):
                    s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                if s != 1:
                    return False
    return True


def _to_int_or_half(vals):
    """Ensure all entries are integer or half-integer; return None if not."""
    for row in vals:
        for x in row:
            f = x * 2
            if abs(f - round(f)) > 1e-9:
                return None
    return vals


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _try_remove_redundant(U, V, W, n, m, p):
    """Remove columns that are linearly redundant in the tensor sense.
    A column r is redundant if removing it still leaves the tensor exact,
    which we test by attempting to compensate with the others (here we just
    test direct removal when the column is all-zero in a way that doesn't
    affect the sum). Simpler: try removing each column and re-verify."""
    rank = len(U[0])
    # Try removing each single column
    best = (U, V, W)
    best_rank = rank
    for r in range(rank):
        if _time_left() <= 0:
            break
        U2 = [row[:r] + row[r + 1:] for row in U]
        V2 = [row[:r] + row[r + 1:] for row in V]
        W2 = [row[:r] + row[r + 1:] for row in W]
        if _verify_exact(U2, V2, W2, n, m, p):
            U, V, W = U2, V2, W2
            best_rank = len(U[0])
            # Loop again since indices shifted
            r = 0
            continue
    return U, V, W


def _strassen_pack(n, m, p):
    """Strassen-style: for n=2, pack output rows.
    (A B) with A,B in R^{2x5}. We want A*C and B*C for C in R^{5x5}.
    Standard: compute (A+B)C, AC, BC -> 3 mults for 2 outputs vs 4.
    But here the 'multiplication' is by a 5x5 matrix, so we can use:
      M1 = (A+B) * C
      M2 = A * C
      M3 = B * C
    Then row0 = M2, row1 = M1 - M2. That's 3 rank-25 blocks = rank 75, worse.

    Better Strassen for 2x2 blocks doesn't directly apply since the inner
    dimension is 5. Instead, use the fact that we can share computations
    across the n=2 rows via linear combinations of A's rows.

    For 2 output rows, we can write:
      out0 = sum_j A[0,j] C[j,:]
      out1 = sum_j A[1,j] C[j,:]
    Any rank decomposition must reproduce these. The minimal rank for
    2x5 x 5x5 is at least 5 (since each output row is a 5-dim linear form
    in A, and the two rows share C). Lower bound: the tensor has a
    'slice rank' of at least max(n*m, m*p, p*n) = 25. So rank >= 25.

    We can achieve rank 25 if we can find U (10x25), V(25x25), W(10x25)
    with the tensor identity. This is a linear system in the products.
    """
    return None  # placeholder; handled by search


def _random_search(n, m, p, target_rank, iterations, seed=0):
    """Randomized search for exact rank-target decomposition with
    integer/half-integer entries. Uses the linear system:
      For each (i,j,k): sum_r U[a,r] V[b,r] W[c,r] = 1.
    This is nonlinear. Instead, fix U and V, solve for W linearly, then
    check W entries are integer/half-integer and exact.
    """
    rng = random.Random(seed)
    a_size = n * m
    b_size = m * p
    c_size = p * n

    for it in range(iterations):
        if _time_left() <= 0:
            break
        # Random integer/half-integer U and V with small entries
        def rand_entry():
            # choose from {-2,-1,-0.5,0,0.5,1,2}
            return Fraction(rng.choice([-4, -2, -1, 0, 1, 2, 4])) / 2

        U = [[rand_entry() for _ in range(target_rank)] for _ in range(a_size)]
        V = [[rand_entry() for _ in range(target_rank)] for _ in range(b_size)]

        # Build linear system for W: for each (i,j,k), sum_r U[a,r]V[b,r] W[c,r] = 1
        # Group by c = k*n+i. For fixed c, the equation involves W[c, :].
        # Actually each (i,j,k) gives one equation. Let's build A_system * w = b
        # where w is the flattened W (c_size * target_rank unknowns).
        # Number of equations = n*m*p = 50. Unknowns = c_size * target_rank = 10*target_rank.
        # For target_rank=25: 250 unknowns, 50 equations -> underdetermined.
        # We need a solution where all W entries are integer/half-integer.

        # Build the system
        eqs = []
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    # coefficient for W[c, r] is U[a,r]*V[b,r]
                    coeffs = {}
                    for r in range(target_rank):
                        idx = c * target_rank + r
                        coeffs[idx] = U[a][r] * V[b][r]
                    eqs.append((coeffs, Fraction(1)))

        # Solve: we have 50 equations, 10*target_rank unknowns.
        # Use rational Gaussian elimination to find a particular solution,
        # then check if entries are integer/half-integer.
        # This is expensive; do a simpler check: try to find W by setting
        # most entries to 0 and solving a small subsystem.

        # Alternative: use numpy least squares in float, then round to
        # integer/half-integer and verify exactly.
        A_mat = np.zeros((n * m * p, c_size * target_rank))
        b_vec = np.ones(n * m * p)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    for r in range(target_rank):
                        A_mat[eq_idx, c * target_rank + r] = float(U[a][r]) * float(V[b][r])
                    eq_idx += 1

        try:
            sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except Exception:
            continue

        # Round to integer/half-integer
        W_float = sol.reshape(c_size, target_rank)
        W_rounded = np.round(W_float * 2) / 2

        # Verify exactly
        W_frac = [[Fraction(int(round(W_rounded[c][r] * 2))) / 2
                   for r in range(target_rank)] for c in range(c_size)]
        if _verify_exact(U, V, W_frac, n, m, p):
            return U, V, W_frac
    return None


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 120.0

    n, m, p = DIMENSIONS

    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best = (U, V, W)
    best_rank = len(U[0])

    # Try removing redundant columns
    U, V, W = _try_remove_redundant(U, V, W, n, m, p)
    if len(U[0]) < best_rank:
        best = (U, V, W)
        best_rank = len(U[0])

    # Randomized search for lower ranks, descending from best_rank-1
    # Theoretical lower bound is 25 (slice rank). Try 25, 26, ..., best_rank-1
    # but prioritize the most promising (lowest) first with limited iterations.
    for target in range(25, best_rank):
        if _time_left() <= 5:
            break
        # Limit iterations based on time
        iters = 200 if target <= 30 else 50
        result = _random_search(n, m, p, target, iters, seed=target)
        if result is not None:
            U2, V2, W2 = result
            if _verify_exact(U2, V2, W2, n, m, p):
                best = (U2, V2, W2)
                best_rank = target
                break  # found the lowest, stop

    U, V, W = best

    # Convert to plain Python numbers (int or float for half-integers)
    def convert(mat):
        out = []
        for row in mat:
            new_row = []
            for x in row:
                f = float(x)
                # Represent half-integers as float; ints as int
                if abs(f - round(f)) < 1e-9:
                    new_row.append(int(round(f)))
                else:
                    new_row.append(f)
            out.append(new_row)
        return out

    return {"U": convert(U), "V": convert(V), "W": convert(W)}
