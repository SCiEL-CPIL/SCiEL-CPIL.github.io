"""Matrix multiplication 3x3 x 3x4: rank-minimal exact tensor decomposition.

We search for a decomposition of the MM(3,3,4) tensor of minimal rank.
The tensor has entries T[(i,j),(j,k),(k,i)] = 1 (trace convention W[k*n+i]).
A rank-r decomposition is U (n*m x r), V (m*p x r), W (p*n x r) with
sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for every (i,j,k).
Entries must be integer or half-integer.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

_DEADLINE = time.time() + 110.0   # safety margin under the 120 s limit


def _within_budget():
    return time.time() < _DEADLINE


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the tensor identity and integer/half-integer entries."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    if V.shape != (m * p, r) or W.shape != (p * n, r):
        return False
    # integer / half-integer check
    for M in (U, V, W):
        doubled = 2.0 * M
        if not np.allclose(doubled, np.round(doubled), atol=tol):
            return False
    # tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fractions."""
    r = U.shape[1]
    if V.shape != (m * p, r) or W.shape != (p * n, r):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _to_fraction_matrix(M):
    return [[Fraction(int(x * 2)) / 2 for x in row] for row in np.asarray(M)]

def _error(U, V, W, n, m, p):
    """Calculate sum of squared errors using Fractions for exactness."""
    err = Fraction(0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                diff = s - 1
                err += diff * diff
    return err

def _local_search_exact(U, V, W, n, m, p, max_iters=1000):
    """
    Perform local search on integer/half-integer grid to minimize error.
    Returns the best (U, V, W) and error found.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    n_rows_u, r = U[0], U[0] if False else len(U[0])
    n_rows_v = len(V)
    n_rows_w = len(W)
    
    current_err = _error(U, V, W, n, m, p)
    
    for _ in range(max_iters):
        if not _within_budget():
            break
        improved = False
        # Try updating U
        for i in range(len(U)):
            for c in range(r):
                for delta in [-1, -1/2, 1/2, 1]:
                    old = U[i][c]
                    U[i][c] = old + delta
                    new_err = _error(U, V, W, n, m, p)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                        break
                    else:
                        U[i][c] = old
        if improved:
            continue
        # Try updating V
        for i in range(len(V)):
            for c in range(r):
                for delta in [-1, -1/2, 1/2, 1]:
                    old = V[i][c]
                    V[i][c] = old + delta
                    new_err = _error(U, V, W, n, m, p)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                        break
                    else:
                        V[i][c] = old
        if improved:
            continue
        # Try updating W
        for i in range(len(W)):
            for c in range(r):
                for delta in [-1, -1/2, 1/2, 1]:
                    old = W[i][c]
                    W[i][c] = old + delta
                    new_err = _error(U, V, W, n, m, p)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                        break
                    else:
                        W[i][c] = old
        if not improved:
            break
            
    return U, V, W, current_err

def _count_rank(U, V, W):
    """Count the number of non-zero columns."""
    r = U[0] if False else len(U[0])
    rank = 0
    for c in range(r):
        if any(U[i][c] != 0 for i in range(len(U))) or \
           any(V[i][c] != 0 for i in range(len(V))) or \
           any(W[i][c] != 0 for i in range(len(W))):
            rank += 1
    return rank

def construct():
    n, m, p = DIMENSIONS
    # Baseline: schoolbook rank 36.
    U0, V0, W0 = _schoolbook(n, m, p)
    U0_frac = _to_fraction_matrix(U0)
    V0_frac = _to_fraction_matrix(V0)
    W0_frac = _to_fraction_matrix(W0)
    
    best_rank = 36
    best_U = [[float(x) for x in row] for row in U0]
    best_V = [[float(x) for x in row] for row in V0]
    best_W = [[float(x) for x in row] for row in W0]

    # Try to find a solution with width < 36 that has lower effective rank
    # We try widths 35, 34, ... down to 20
    for width in range(35, 19, -1):
        if not _within_budget():
            break
        # Initialize with random small integers
        rng = np.random.default_rng(width)
        for attempt in range(20):
            if not _within_budget():
                break
            U = rng.integers(-2, 3, size=(n*m, width))
            V = rng.integers(-2, 3, size=(m*p, width))
            W = rng.integers(-2, 3, size=(p*n, width))
            
            U_frac = _to_fraction_matrix(U)
            V_frac = _to_fraction_matrix(V)
            W_frac = _to_fraction_matrix(W)
            
            # Local search
            U_loc, V_loc, W_loc, err = _local_search_exact(U_frac, V_frac, W_frac, n, m, p, max_iters=200)
            
            if err == 0:
                rank = _count_rank(U_loc, V_loc, W_loc)
                if rank < best_rank:
                    best_rank = rank
                    best_U = [[float(x) for x in row] for row in U_loc]
                    best_V = [[float(x) for x in row] for row in V_loc]
                    best_W = [[float(x) for x in row] for row in W_loc]
                    if best_rank <= 23: # Unlikely but good to know
                        break
            # Also check if the local search reduced error significantly, maybe it's close?
            # But we need exact 0.

    return {"U": best_U, "V": best_V, "W": best_W}
