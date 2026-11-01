"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import sys
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(result - T)) < tol

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

def _optimize(U, V, W, n, m, p, max_iters=2000, lr=0.001):
    T = _build_tensor(n, m, p)
    U = U.copy()
    V = V.copy()
    W = W.copy()
    for it in range(max_iters):
        result = np.einsum('ar,br,cr->abc', U, V, W)
        err = result - T
        loss = np.sum(err**2)
        if loss < 1e-10:
            break
        gU = 2*np.einsum('abc,br,cr->ar', err, V, W)
        gV = 2*np.einsum('abc,ar,cr->br', err, U, W)
        gW = 2*np.einsum('abc,ar,br->cr', err, U, V)
        U -= lr*gU
        V -= lr*gV
        W -= lr*gW
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions via optimization
    for target_rank in range(31, best_rank):
        if time.time() > deadline - 25:
            break
        for attempt in range(3):
            if time.time() > deadline - 25:
                break
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            U, V, W = _optimize(U, V, W, n, m, p, max_iters=1500, lr=0.001)
            
            # Round to half-integers and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-9):
                best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = target_rank
                break
    
    return best

# EVOLVE-BLOCK-END
