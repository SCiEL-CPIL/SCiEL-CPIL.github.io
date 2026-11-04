"""Matrix multiplication 2x3 x 3x4 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# Tensor T[i,j,k] = 1 if output index (i,k) is produced by U[i,j] * V[j,k]
# We want U (n*m x r), V (m*p x r), W (p*n x r) such that
# for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
# This is the standard matrix multiplication tensor decomposition.

def _make_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3D array T[i,j,k] = 1."""
    T = np.ones((n, m, p))
    return T

def _tensor_residual(U, V, W, n, m, p):
    """Compute max absolute residual over all (i,j,k)."""
    r = U.shape[1]
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Residual[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
    maxres = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                res = abs(s - 1.0)
                if res > maxres:
                    maxres = res
    return maxres

def _snap_to_rationals(M, max_den=8):
    """Snap each entry to the nearest fraction with denominator <= max_den."""
    r, c = M.shape
    out = np.zeros_like(M, dtype=object)
    for a in range(r):
        for b in range(c):
            x = float(M[a, b])
            best = Fraction(0)
            best_err = abs(x)
            for d in range(1, max_den + 1):
                num = round(x * d)
                fr = Fraction(num, d)
                err = abs(x - float(fr))
                if err < best_err:
                    best_err = err
                    best = fr
            out[a, b] = best
    return out

def _verify_exact(U, V, W, n, m, p):
    """Return True if the decomposition is exact (all (i,j,k) give exactly 1)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for rr in range(r):
                    s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                if s != 1:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iter=4000, tol=1e-12, seed=0):
    """Alternating least squares to find U,V,W of given rank."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1.0, (n*m, rank))
    V = rng.normal(0, 1.0, (m*p, rank))
    W = rng.normal(0, 1.0, (p*n, rank))
    # Normalize initial scale
    for _ in range(max_iter):
        # Update U: for each row (i*m+j), solve least squares
        # U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:]) over k = 1
        # Actually the constraint couples all (i,j,k). Use gradient-free ALS on the
        # "flattened" view is non-trivial; instead do a simple projected gradient step.
        # Use a simple multiplicative/gradient approach on the log of magnitudes.
        # Simpler: do coordinate descent via least squares per row using the
        # "contracted" matrices.
        # Build A_U[i*m+j, r] = sum_k V[j*p+k,r]*W[k*n+i,r]
        # Then U[i*m+j,:] should approximate row of target where target=1 for all k,
        # but the same U row is used for all k. So we want U row * A_U^T = 1 for all k.
        # This is: U_row @ A_U[:, :].T = ones(p). Solve least squares.
        A_U = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                for rr in range(rank):
                    s = 0.0
                    for k in range(p):
                        s += V[j*p+k, rr] * W[k*n+i, rr]
                    A_U[i*m+j, rr] = s
        for i in range(n):
            for j in range(m):
                A = A_U[i*m+j, :].reshape(1, rank)
                # minimize ||U_row @ A.T - 1||^2  => U_row = (A A^T)^{-1} A
                # U_row is (1 x rank): U_row = A / (A @ A.T)
                denom = A @ A.T
                if denom[0,0] < 1e-15:
                    continue
                U[i*m+j, :] = A / denom
        # Update V similarly
        A_V = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                for rr in range(rank):
                    s = 0.0
                    for i in range(n):
                        s += U[i*m+j, rr] * W[k*n+i, rr]
                    A_V[j*p+k, rr] = s
        for j in range(m):
            for k in range(p):
                A = A_V[j*p+k, :].reshape(1, rank)
                denom = A @ A.T
                if denom[0,0] < 1e-15:
                    continue
                V[j*p+k, :] = A / denom
        # Update W similarly
        A_W = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for rr in range(rank):
                    s = 0.0
                    for j in range(m):
                        s += U[i*m+j, rr] * V[j*p+k, rr]
                    A_W[k*n+i, rr] = s
        for k in range(p):
            for i in range(n):
                A = A_W[k*n+i, :].reshape(1, rank)
                denom = A @ A.T
                if denom[0,0] < 1e-15:
                    continue
                W[k*n+i, :] = A / denom
        res = _tensor_residual(U, V, W, n, m, p)
        if res < tol:
            break
    return U, V, W, res

def _try_rank(n, m, p, rank, seed=0, max_den=8):
    """Try to find an exact decomposition of given rank. Return (U,V,W) as Fraction arrays or None."""
    U, V, W, res = _als_decompose(n, m, p, rank, max_iter=3000, tol=1e-13, seed=seed)
    if res > 1e-6:
        return None
    # Snap to rationals
    Uf = _snap_to_rationals(U, max_den)
    Vf = _snap_to_rationals(V, max_den)
    Wf = _snap_to_rationals(W, max_den)
    if _verify_exact(Uf, Vf, Wf, n, m, p):
        return Uf, Vf, Wf
    # Try larger denominators
    for d in (16, 32, 64):
        Uf = _snap_to_rationals(U, d)
        Vf = _snap_to_rationals(V, d)
        Wf = _snap_to_rationals(W, d)
        if _verify_exact(Uf, Vf, Wf, n, m, p):
            return Uf, Vf, Wf
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0  # keep margin before 150s kill
    best = None
    # Try ranks from low to high
    for rank in range(12, 25):
        if time.time() > deadline - 20.0:
            break
        for seed in range(8):
            if time.time() > deadline - 10.0:
                break
            res = _try_rank(n, m, p, rank, seed=seed)
            if res is not None:
                Uf, Vf, Wf = res
                best = (rank, Uf, Vf, Wf)
                break
        if best is not None:
            break
    if best is None:
        # Fallback: trivial rank-24 decomposition (guaranteed exact)
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return {"U": U, "V": V, "W": W}
    rank, Uf, Vf, Wf = best
    # Convert Fraction arrays to lists of Fractions (or ints) for the interface
    def to_list(M):
        r, c = M.shape
        out = []
        for a in range(r):
            row = []
            for b in range(c):
                v = M[a, b]
                if isinstance(v, Fraction):
                    if v.denominator == 1:
                        row.append(int(v.numerator))
                    else:
                        row.append(v)
                else:
                    row.append(int(round(float(v))))
            out.append(row)
        return out
    return {"U": to_list(Uf), "V": to_list(Vf), "W": to_list(Wf)}
