"""
Rank-minimal exact tensor decomposition for (2,3,8) matrix multiplication.

We search for a rank-r decomposition using random rank-r initialisations and
gradient descent on a "completion" formulation: fix a subset of r indices for
each tensor, and optimise the free entries so that the tensor equation is
satisfied exactly (or to machine precision).

The output follows the trace convention W[k*n + i, r].
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 48

# Target tensor T[i, j, k] = 1 for all (i, j, k)
# The decomposition must satisfy:
#   T[i, j, k] = sum_c U[i*M + j, c] * V[j*P + k, c] * W[k*N + i, c]

def _make_target_tensor():
    T = np.zeros((N, M, P), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-7):
    """Check if the decomposition exactly reproduces T."""
    r = U.shape[1]
    C = np.zeros((N, M, P), dtype=float)
    for c in range(r):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(N):
            for j in range(M):
                uval = u[i * M + j]
                if abs(uval) < 1e-14:
                    continue
                for k in range(P):
                    C[i, j, k] += uval * v[j * P + k] * w[k * N + i]
    return np.max(np.abs(C - T)) < tol

def _try_decompose(r, T, max_iters=2000, lr=0.01, seed=0, time_budget=8.0):
    """
    Try to find a rank-r decomposition using gradient descent.
    We parameterise U, V, W freely (all r columns) and minimise
    sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - T[i,j,k])^2
    """
    rng = np.random.RandomState(seed)
    
    # Initial guess: small random values
    U = rng.randn(N * M, r) * 0.1
    V = rng.randn(M * P, r) * 0.1
    W = rng.randn(P * N, r) * 0.1
    
    # We want the sum to equal 1. Let's try to make at least one term contribute.
    # Actually, let's use a smarter init: set some entries to 1 and others to 0
    # and see if gradient descent can find a solution.
    
    # Alternative: use the "completion" idea. Pick r random (i,j,k) triples,
    # and try to set U,V,W so that those triples give 1 and others give 0.
    # This is underdetermined, so we need to be careful.
    
    # Let's just do gradient descent on the full objective.
    start = time.time()
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute C = sum_c outer(U[:,c], V[:,c], W[:,c]) reshaped
        C = np.zeros((N, M, P), dtype=float)
        for c in range(r):
            u = U[:, c].reshape(N, M)
            v = V[:, c].reshape(M, P)
            w = W[:, c].reshape(P, N)
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        C[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        err = C - T
        obj = 0.5 * np.sum(err ** 2)
        if obj < 1e-14:
            return U, V, W
        
        # Compute gradients
        # d_obj/dU[i,j,c] = sum_{k} err[i,j,k] * V[j,k,c] * W[k,i,c]
        # d_obj/dV[j,k,c] = sum_{i} err[i,j,k] * U[i,j,c] * W[k,i,c]
        # d_obj/dW[k,i,c] = sum_{j} err[i,j,k] * U[i,j,c] * V[j,k,c]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        e = err[i, j, k]
                        dU[i * M + j, c] += e * V[j * P + k, c] * W[k * N + i, c]
                        dV[j * P + k, c] += e * U[i * M + j, c] * W[k * N + i, c]
                        dW[k * N + i, c] += e * U[i * M + j, c] * V[j * P + k, c]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate
        if it % 100 == 0:
            new_obj = 0.5 * np.sum(err ** 2)
            # Simple: if obj not decreasing, reduce lr
            pass
    
    if _verify(U, V, W, T):
        return U, V, W
    return None

def _schoolbook():
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    T = _make_target_tensor()
    
    # Try to find a decomposition with rank < 48
    # Start from 47 down to 40 (or as far as time allows)
    for r in range(RANK_SCHOOLBOOK - 1, 39, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            result = _try_decompose(r, T, max_iters=1500, lr=0.02, seed=seed,
                                     time_budget=min(remaining - 0.5, 6.0))
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    return _schoolbook()
