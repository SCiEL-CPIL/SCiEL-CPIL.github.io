"""Rewritten program: search for a low-rank exact tensor decomposition of 2x3x3 matmul."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=float)
    return T

def _decompose_error(U, V, W):
    """Compute the error of the decomposition."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j, c] * V[j*p+l, c] * W[l*n+i, c]
                err += (s - 1.0)**2
    return err

def _to_frac(x, tol=1e-6):
    """Snap to a simple rational if close enough."""
    if abs(x) < tol:
        return 0.0
    # Try integers and half-integers first
    for den in [1, 2, 3, 4, 5, 6]:
        num = round(x * den)
        if abs(x - num/den) < tol:
            if num == 0:
                return 0.0
            return float(num) / den
    return x

def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = Fraction(0, 1)
                for c in range(R):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+l][c]) * Fraction(W[l*n+i][c])
                if s != 1:
                    return False
    return True

def _snap_and_verify(U, V, W):
    """Try to snap floating point values to simple rationals and verify."""
    U_r = [[_to_frac(x) for x in row] for row in U]
    V_r = [[_to_frac(x) for x in row] for row in V]
    W_r = [[_to_frac(x) for x in row] for row in W]
    if _verify_exact(U_r, V_r, W_r):
        return (U_r, V_r, W_r)
    return None

def _coordinate_descent(U, V, W, n_inner=30):
    """Perform coordinate descent to minimize decomposition error."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    for _ in range(n_inner):
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i*m+j
            M = np.zeros((p, R))
            b = np.ones(p)
            for l in range(p):
                for c in range(R):
                    M[l, c] = V[j*p+l, c] * W[l*n+i, c]
            try:
                sol = np.linalg.lstsq(M, b, rcond=None)[0]
                U[idx, :] = sol
            except:
                pass
        # Update V
        for j in range(m):
            for l in range(p):
                idx = j*p+l
                M = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    for c in range(R):
                        M[i, c] = U[i*m+j, c] * W[l*n+i, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
        # Update W
        for l in range(p):
            for i in range(n):
                idx = l*n+i
                M = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    for c in range(R):
                        M[j, c] = U[i*m+j, c] * V[j*p+l, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
    return U, V, W

def _try_rank_R(R, n_seeds=5, max_time=10.0, seed_base=0):
    """Try to find a rank-R decomposition."""
    n, m, p = DIMENSIONS
    deadline = time.time() + max_time
    best_err = float('inf')
    best = None
    
    for seed in range(seed_base, seed_base + n_seeds):
        if time.time() > deadline - 2:
            break
        rng = np.random.RandomState(seed)
        # Initialize with small random values
        U = rng.randn(n*m, R) * 0.5
        V = rng.randn(m*p, R) * 0.5
        W = rng.randn(p*n, R) * 0.5
        
        # Run coordinate descent
        U, V, W = _coordinate_descent(U, V, W, n_inner=40)
        
        err = _decompose_error(U, V, W)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-10:
            break
    
    if best is None:
        return None
    
    if best_err < 1e-8:
        result = _snap_and_verify(*best)
        if result is not None:
            return result
    
    return None

def _try_structured_rank(R):
    """Try structured initialization based on known patterns."""
    n, m, p = DIMENSIONS
    # For 2x3x3, try various structured initializations
    patterns = []
    
    # Pattern 1: block diagonal style
    for offset in range(R):
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        # Assign each column to a different (i,j,l) combination
        idx = offset % (n*m*p)
        i = idx // (m*p)
        j = (idx // p) % m
        l = idx % p
        U[i*m+j, offset] = 1.0
        V[j*p+l, offset] = 1.0
        W[l*n+i, offset] = 1.0
        patterns.append((U.copy(), V.copy(), W.copy()))
    
    best_err = float('inf')
    best = None
    for U, V, W in patterns:
        U, V, W = _coordinate_descent(U, V, W, n_inner=20)
        err = _decompose_error(U, V, W)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
    
    if best is not None and best_err < 1e-8:
        result = _snap_and_verify(*best)
        if result is not None:
            return result
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
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
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try ranks from 12 up to 17 (skip 18 schoolbook)
    for R in range(12, 18):
        if time.time() > deadline - 15:
            break
        
        # First try structured initialization
        if time.time() < deadline - 20:
            res = _try_structured_rank(R)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
        
        # Then try random coordinate descent
        for seed_base in [0, 100, 200, 300]:
            if time.time() > deadline - 10:
                break
            res = _try_rank_R(R, n_seeds=3, max_time=5.0, seed_base=seed_base)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
