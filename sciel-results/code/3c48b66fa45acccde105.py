"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]
DEADLINE_MARGIN = 5.0  # seconds to spare before the 150s kill
EVAL_START = time.time()

# EVOLVE-BLOCK-START

def _time_left():
    """Return seconds remaining before the hard deadline."""
    return (150.0 - DEADLINE_MARGIN) - (time.time() - EVAL_START)

def _schoolbook():
    """Return the rank-48 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=4, p=4):
    """Check exact decomposition on random matrices. Returns True if valid."""
    rank = len(V[0]) if V else 0
    for trial in range(3):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C = A @ B
        # U shape: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Ua = np.array(U, dtype=float).reshape(n, m, rank)
        Va = np.array(V, dtype=float).reshape(m, p, rank)
        Wa = np.array(W, dtype=float).reshape(p, n, rank)
        C_recon = np.einsum('ijk,akl,lbm->abm', Ua, Va, Wa)
        if not np.allclose(C_recon, C, atol=1e-9):
            return False
    return True

def _try_rank(R, seed=None):
    """Try to find a rank-R decomposition via random initialization + local search."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    n, m, p = DIMENSIONS
    # Initialize with schoolbook, then randomly permute and zero out R-n*m*p columns
    if R >= n * m * p:
        return _schoolbook()
    # Random initialization: pick R random tensors
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    for r in range(R):
        U[:, r] = np.random.uniform(-1, 1, n * m)
        V[:, r] = np.random.uniform(-1, 1, m * p)
        W[:, r] = np.random.uniform(-1, 1, p * n)
    # Use a simple iterative method: minimize ||T - sum U_r ⊗ V_r ⊗ W_r||^2
    # by alternating projections (similar to CP decomposition but for 3-way tensor)
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.inf
    lr = 0.01
    for it in range(200):
        if _time_left() < 10.0:
            break
        Ua = U.reshape(n, m, R)
        Va = V.reshape(m, p, R)
        Wa = W.reshape(p, n, R)
        C_recon = np.einsum('ijk,akl,lbm->abm', Ua, Va, Wa)
        err = np.linalg.norm(C_recon - T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-8:
                break
        # Gradient step
        # d/dU: for each a,b,r: sum_k Wa[k,a,r] * Va[b,k,r] * (C_recon[a,b] - T[a,b])
        grad_U = np.einsum('kam,bkm,ab->abm', Wa, Va, C_recon - T)
        grad_V = np.einsum('kam,bkm,ab->bkm', Wa, Ua, C_recon - T)
        grad_W = np.einsum('kam,bkm,ab->kam', Ua, Va, C_recon - T)
        U -= lr * grad_U.reshape(n * m, R)
        V -= lr * grad_V.reshape(m * p, R)
        W -= lr * grad_W.reshape(p * n, R)
    # Check best
    if best_err < 1e-7:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None

def construct():
    deadline = 150.0 - DEADLINE_MARGIN
    # Try a few random ranks below 48
    best = None
    best_rank = 48
    school = _schoolbook()
    best = school
    best_rank = 48
    for R in range(35, 48):
        if _time_left() < 15.0:
            break
        result = _try_rank(R, seed=R * 12345)
        if result is not None:
            # Verify
            if _verify(result["U"], result["V"], result["W"]):
                best = result
                best_rank = R
                break
    return best
# EVOLVE-BLOCK-END
