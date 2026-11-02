"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
import numpy as np
import time
import sys
import random

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    # Build the multiplication tensor T[i, j, k] = 1 (schoolbook)
    # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    rank = U.shape[1]
    # Efficient check: for each c, the rank-1 tensor is outer of U_col, V_col, W_col
    # We'll check a few random points and all points if small
    # For n=2,m=3,p=7, total points = 2*3*7=42
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _try_rank(r, n, m, p, max_iters=200, seed=42):
    """Try to find a rank-r decomposition using numerical optimization."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize with random matrices
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    # Build the target tensor T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    best_loss = float('inf')
    lr = 0.01
    
    for it in range(max_iters):
        # Compute current tensor
        # T_current[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        T_cur = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            # T_cur[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_cur[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        
        loss = np.sum((T_cur - T) ** 2)
        if loss < best_loss:
            best_loss = loss
            if best_loss < 1e-10:
                break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 2 * (T_cur[i, j, k] - T[i, j, k])
                        dU[i * m + j, c] += err * vc[j, k] * wc[k, i]
                        dV[j * p + k, c] += err * uc[i, j] * wc[k, i]
                        dW[k * n + i, c] += err * uc[i, j] * vc[j, k]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Final verification
    if best_loss < 1e-8:
        return U, V, W, best_loss
    return None, None, None, best_loss

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Leave 5s margin
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also, for matrix multiplication, rank >= max(n*m, m*p, p*n) is a lower bound
    # But there might be better decompositions
    
    # Try ranks from 21 up to best_rank-1
    for r in range(21, best_rank):
        if time.time() > deadline - 30:
            break
        # Try a few seeds
        for seed in [42, 123, 456, 789, 101112]:
            if time.time() > deadline - 30:
                break
            U, V, W, loss = _try_rank(r, n, m, p, max_iters=100, seed=seed)
            if U is not None and loss < 1e-8:
                # Verify with higher precision
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
