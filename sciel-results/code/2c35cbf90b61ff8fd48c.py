"""Rewritten program for matrix multiplication 3,4,5 tensor rank minimization."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    rank = n*m*p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """Compute the max absolute error of the tensor reconstruction."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] = 1.0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i*m + j, :]
                vj = V[j*p + k, :]
                wk = W[k*n + i, :]
                val = np.dot(ui, vj * wk)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _gradient_step(U, V, W, n, m, p, lr):
    """One gradient descent step to minimize sum of (T[i,j,k]-1)^2."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    # Compute gradients analytically
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # d/dU[i*m+j, c] = sum over (j,k) where this U is used: (T[i,j,k]-1) * V[j*p+k, c] * W[k*n+i, c]
    # Actually, for a fixed (i,j), U[i*m+j, c] appears in T[i,j,k] for all k.
    # So grad_U[i*m+j, c] = sum_k (T[i,j,k]-1) * V[j*p+k, c] * W[k*n+i, c]
    
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i*m + j, :]
                vj = V[j*p + k, :]
                wk = W[k*n + i, :]
                val = np.dot(ui, vj * wk)
                diff = val - 1.0
                # grad U[i*m+j, c] += diff * V[j*p+k, c] * W[k*n+i, c]
                gU[i*m + j, :] += diff * (vj * wk)
                # grad V[j*p+k, c] += diff * U[i*m+j, c] * W[k*n+i, c]
                gV[j*p + k, :] += diff * (ui * wk)
                # grad W[k*n+i, c] += diff * U[i*m+j, c] * V[j*p+k, c]
                gW[k*n + i, :] += diff * (ui * vj)
    
    U_new = U - lr * gU
    V_new = V - lr * gV
    W_new = W - lr * gW
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    deadline = time.time() + 110  # Leave 10s margin
    
    # Schoolbook decomposition
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    rank_school = n * m * p
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: For each rank r from rank_school-1 down, try to find a valid decomposition
    # using gradient descent from various initializations.
    # Key insight: initialize near schoolbook by randomly merging/removing terms.
    
    for r in range(rank_school - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        # Number of trials: more for higher ranks where improvement is more likely
        num_trials = 15 if r >= 55 else 8 if r >= 50 else 4
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Initialization strategy: start from schoolbook and perturb
            # Create a rank-r initialization by taking random linear combinations of schoolbook terms
            # This is equivalent to: pick r random directions in the 60-dim schoolbook space
            
            if r < rank_school:
                # Create random mixing matrix M of shape (rank_school, r)
                # Then U_new = U_school @ M, etc.
                # But we need the result to still approximate the tensor.
                # Actually, this won't work directly because the tensor is a product.
                
                # Better: random Gaussian initialization with moderate scale
                scale = 0.5
                U = np.random.randn(nm, r) * scale
                V = np.random.randn(mp, r) * scale
                W = np.random.randn(pn, r) * scale
            else:
                U = U_school.copy()
                V = V_school.copy()
                W = W_school.copy()
            
            # Run gradient descent
            lr = 0.01
            converged = False
            for step in range(200):
                if time.time() > deadline:
                    break
                
                err = _tensor_error(U, V, W, n, m, p)
                if err < 1e-8:
                    converged = True
                    break
                
                # Adaptive learning rate
                U, V, W = _gradient_step(U, V, W, n, m, p, lr)
                
                # Reduce learning rate periodically
                if step % 50 == 49:
                    lr *= 0.5
            
            if converged:
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # Found a valid decomposition at this rank; try lower ranks
                break
    
    # Final polish: if we found a solution, try to verify with higher precision
    # and potentially refine
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
