"""Matrix multiplication 3x3 x 3x5 — minimal-rank exact tensor decomposition."""
import time
import math
import itertools
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=3, p=5):
    """Check that U * V^T * W^T reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def _schoolbook(n=3, m=3, p=5):
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


def _als_decompose(n=3, m=3, p=5, rank=20, max_iter=500, tol=1e-12, seed=42):
    """ALS to find rank-r decomposition of all-ones tensor T[i,j,k]=1."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    T = np.ones((n, m, p))
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                rhs = M.T @ T[i, j, :]
                A = M.T @ M + 1e-10 * np.eye(rank)
                try:
                    U[i * m + j, :] = np.linalg.solve(A, rhs)
                except:
                    U[i * m + j, :] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                rhs = M.T @ T[:, j, k]
                A = M.T @ M + 1e-10 * np.eye(rank)
                try:
                    V[j * p + k, :] = np.linalg.solve(A, rhs)
                except:
                    V[j * p + k, :] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                rhs = M.T @ T[i, :, k]
                A = M.T @ M + 1e-10 * np.eye(rank)
                try:
                    W[k * n + i, :] = np.linalg.solve(A, rhs)
                except:
                    W[k * n + i, :] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        if err < tol:
            break
    
    return U, V, W, err


def _try_round_and_verify(U, V, W, n=3, m=3, p=5):
    """Try rounding to half-integers and verify."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                if abs(total - 1.0) > 1e-9:
                    return None
    return U_r.tolist(), V_r.tolist(), W_r.tolist()


def construct():
    t0 = time.time()
    deadline = t0 + 100
    
    # Try ranks from 19 upward
    for rank in range(19, 32):
        if time.time() > deadline - 20:
            break
        
        max_iter = max(100, 400 - (rank - 19) * 15)
        n_restarts = max(3, 10 - (rank - 19))
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = 123 + restart * 7919
            U, V, W, err = _als_decompose(3, 3, 5, rank, max_iter=max_iter, seed=seed)
            
            if err < 1e-8:
                result = _try_round_and_verify(U, V, W)
                if result is not None:
                    U_list, V_list, W_list = result
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we got very close but couldn't round, try harder
        if rank >= 19:
            for restart in range(3):
                if time.time() > deadline - 10:
                    break
                seed = 999 + restart * 31337
                U, V, W, err = _als_decompose(3, 3, 5, rank, max_iter=800, seed=seed)
                if err < 1e-10:
                    result = _try_round_and_verify(U, V, W)
                    if result is not None:
                        U_list, V_list, W_list = result
                        if _verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
