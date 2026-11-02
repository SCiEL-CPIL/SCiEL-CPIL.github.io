"""Tensor decomposition for 3x3 * 3x4 matrix multiplication.
Aim: exact decomposition of rank < 36.  Strategy: start from the schoolbook
tensor, then run an iterative rank-reduction / completion scheme over
integer/half-integer entries, verifying exactness each step.
"""
DIMENSIONS = [3, 3, 4]

import time
import random
import numpy as np
from fractions import Fraction

_DEADLINE = 130.0  # seconds from start of construct
_start_t = None


def _now():
    global _start_t
    if _start_t is None:
        _start_t = time.time()
    return time.time()


def _remaining():
    return _DEADLINE - _now()


def _build_target():
    """Return the multiplication tensor as a 3D array of Fraction.
    Indexing: T[i * m + j, k * n + i?] -- but the decomposition is defined by
    (U_{i j}, V_{j k}, W_{k i}) for each product.  We represent the target
    bilinear form: for each pair (row of A, col of B), the coefficient of
    intermediate r that must appear is 1.
    Here we use a flat 2D representation:
      A_rows = n*m = 9
      B_cols = m*p = 12
      C_pairs = p*n = 12
    Target tensor is a 3-way array of shape (9,12,12) where T[a,b,c] = 1 iff
    c = k*n + i and a = i*m + j and b = j*p + k for some (i,j,k).
    """
    n, m, p = DIMENSIONS
    ar, bc, cp = n * m, m * p, p * n
    T = np.zeros((ar, bc, cp), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = Fraction(1)
    return T, ar, bc, cp


def _verify(U, V, W, T):
    """Check that sum_r U[:,r] outer V[:,r] * W[:,r] == T exactly."""
    ar, bc, cp = T.shape
    rank = U.shape[1]
    R = np.zeros((ar, bc, cp), dtype=object)
    for r in range(rank):
        R += (np.outer(U[:, r], V[:, r]))[:, :, np.newaxis] * W[:, r]
    # Compare with T
    return np.all(R == T)


def _rank_one_decompose(T, ar, bc, cp, deadline):
    """Greedy search for a small exact decomposition with small rational entries.
    We try rank R from 22 up to 35.  For each R we attempt a randomized
    integer/half-integer completion with iterative refinement.
    """
    best = None
    for R in range(22, 36):
        if _remaining() < 8:
            break
        # Try multiple seeds
        for seed in range(4):
            if _remaining() < 8:
                break
            rng = random.Random(seed * 1000 + R)
            U, V, W = _attempt_rank(R, T, ar, bc, cp, rng, _remaining() - 12)
            if U is not None:
                if _verify(U, V, W, T):
                    if best is None or R < best[0]:
                        best = (R, U, V, W)
        if best is not None:
            return best
    return best


def _attempt_rank(R, T, ar, bc, cp, rng, time_budget):
    """Try to find U,V,W of given rank with small half-integer entries.
    Method: parameterize all entries as Fraction with denominator 2.
    Use a simple local search: pick a random tensor, then greedily set entries
    to reduce the error, verifying exactness.
    Since the problem is small (9x12x12, R<=35 -> at most 9*R + 12*R + 12*R = 33R <= 1155 vars)
    we can attempt a randomized local search over a subset.
    """
    # Initialize with schoolbook (rank 36) and try to zero out some rows/columns.
    # A simpler idea: start with the tensor expressed in a basis of rank-R tensors.
    # We use a random decomposition and then adjust.
    # Given the complexity and time constraints, we instead try a structured construction:
    # Use the 3x3*3x3 Strassen-like core (if available) and extend.
    # For now, use a random initialization and local refinement.
    
    def _rand_halfint():
        return rng.choice([-2, -1, 0, 1, 2]) / 2  # will be Fraction
    
    U = [[Fraction(_rand_halfint()) for _ in range(R)] for _ in range(ar)]
    V = [[Fraction(_rand_halfint()) for _ in range(R)] for _ in range(bc)]
    W = [[Fraction(_rand_halfint()) for _ in range(R)] for _ in range(cp)]
    
    def _error():
        R3 = np.zeros((ar, bc, cp), dtype=object)
        for r in range(R):
            R3 += (np.outer(U[:, r], V[:, r]))[:, :, np.newaxis] * W[:, r]
        return np.sum(1 for i in range(ar) for j in range(bc) for k in range(cp) if R3[i,j,k] != T[i,j,k])
    
    err0 = _error()
    if err0 == 0:
        return U, V, W
    
    # Local search: repeatedly pick a random variable and set it to a value that minimizes local error.
    # For speed, we only evaluate the affected slice.
    deadline_local = _remaining() - 12
    step = 0
    while _remaining() > deadline_local + 5:
        step += 1
        if step % 500 == 0 and _error() == 0:
            return U, V, W
        
        # Pick a random matrix and entry
        which = rng.choice([0, 1, 2])
        if which == 0:
            a = rng.randrange(ar)
            r = rng.randrange(R)
            # Try values for U[a, r]
            best_val = None
            best_err = _error()
            for val in [Fraction(x, 2) for x in range(-4, 5)]:
                if U[a][r] == val:
                    continue
                old = U[a][r]
                U[a][r] = val
                err = _error()
                if err < best_err:
                    best_err = err
                    best_val = val
                U[a][r] = old
            if best_val is not None:
                U[a][r] = best_val
        elif which == 1:
            b = rng.randrange(bc)
            r = rng.randrange(R)
            best_val = None
            best_err = _error()
            for val in [Fraction(x, 2) for x in range(-4, 5)]:
                if V[b][r] == val:
                    continue
                old = V[b][r]
                V[b][r] = val
                err = _error()
                if err < best_err:
                    best_err = err
                    best_val = val
                V[b][r] = old
            if best_val is not None:
                V[b][r] = best_val
        else:
            c = rng.randrange(cp)
            r = rng.randrange(R)
            best_val = None
            best_err = _error()
            for val in [Fraction(x, 2) for x in range(-4, 5)]:
                if W[c][r] == val:
                    continue
                old = W[c][r]
                W[c][r] = val
                err = _error()
                if err < best_err:
                    best_err = err
                    best_val = val
                W[c][r] = old
            if best_val is not None:
                W[c][r] = best_val
    
    if _error() == 0:
        return U, V, W
    return None


# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the 3x3 * 3x4 matrix multiplication tensor.
    """
    global _start_t
    _start_t = time.time()
    _T = _build_target()
    T, ar, bc, cp = _T
    
    # First, fallback: schoolbook rank 36
    def _schoolbook():
        U = [[0] * 36 for _ in range(ar)]
        V = [[0] * 36 for _ in range(bc)]
        W = [[0] * 36 for _ in range(cp)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i * 3 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return U, V, W, 36
    
    best_rank, U, V, W = _schoolbook()
    
    # Attempt to find a lower rank decomposition
    if _remaining() > 20:
        res = _rank_one_decompose(T, ar, bc, cp, _remaining() - 15)
        if res is not None:
            rank, U, V, W = res
            best_rank = rank
    
    # Convert Fraction entries to int or float(half-integer)
    def _conv(mat):
        out = []
        for row in mat:
            newrow = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        newrow.append(int(val))
                    else:
                        newrow.append(float(val))
                else:
                    newrow.append(val)
            out.append(newrow)
        return out
    
    return {"U": _conv(U), "V": _conv(V), "W": _conv(W)}
# EVOLVE-BLOCK-END
