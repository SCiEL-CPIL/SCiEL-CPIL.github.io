"""Tensor decomposition search for 2x5x6 matrix multiplication."""
import numpy as np
import time
import itertools
import sys
import random

DIMENSIONS = [2, 5, 6]
SEED_RANK = 60

def _construct_schoolbook(n=2, m=5, p=6):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank_of(A, B, W, tol=1e-6):
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for d in range(p):
                    if a == i and d == k:
                        for b in range(m):
                            T[i * m + b, b * p + d, k * n + i] = 1.0
    R = np.zeros((n * m, m * p, p * n))
    for c in range(W.shape[1]):
        R += np.einsum('i,j,k->ijk', A[:, c], B[:, c], W[:, c])
    return np.linalg.norm(R - T)

def _verify_exact(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                T[i * m + b, b * p + k, k * n + i] = 1.0
    R = np.zeros((n * m, m * p, p * n))
    for c in range(W.shape[1]):
        R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(R - T)) < tol

def _random_init(r, n=2, m=5, p=6):
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    return U, V, W

def _try_rank(r, time_budget, n=2, m=5, p=6):
    start_time = time.time()
    target_time = start_time + time_budget
    np.random.seed(42)
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for trial in range(200):
        if time.time() > target_time - 1.0:
            break
        
        U, V, W = _random_init(r, n, m, p)
        err = _rank_of(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Local search: perturb one factor at a time
        for _ in range(30):
            if time.time() > target_time - 1.0:
                break
            factor = random.choice([0, 1, 2])
            col = random.randint(0, r - 1)
            if factor == 0:
                U[:, col] += np.random.randn(n * m) * 0.1
                U = np.clip(U, -10, 10)
            elif factor == 1:
                V[:, col] += np.random.randn(m * p) * 0.1
                V = np.clip(V, -10, 10)
            else:
                W[:, col] += np.random.randn(p * n) * 0.1
                W = np.clip(W, -10, 10)
            
            err = _rank_of(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
    
    return best_U, best_V, best_W, best_err

def _try_structured(n=2, m=5, p=6, r=40):
    """Try to exploit the structure: C[i,k] = sum_b A[i,b]*B[b,k]"""
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # First n*m*p=60 terms for schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    deadline = time.time() + 115  # 115s to be safe under 120s
    n, m, p = DIMENSIONS
    schoolbook = _construct_schoolbook(n, m, p)
    best_solution = schoolbook
    best_rank = SEED_RANK
    
    # Try ranks from lower to higher
    for r in range(30, 60):
        if time.time() > deadline - 10:
            break
        time_left = deadline - time.time() - 5
        if time_left < 1.0:
            break
        U, V, W, err = _try_rank(r, time_left, n, m, p)
        if err < 1e-6 and _verify_exact(U, V, W, 1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if err < 1e-3:
            # Close but not exact; keep as backup if better than schoolbook
            pass
    
    # Fallback to schoolbook
    return {
        "U": schoolbook["U"].tolist(),
        "V": schoolbook["V"].tolist(),
        "W": schoolbook["W"].tolist()
    }
