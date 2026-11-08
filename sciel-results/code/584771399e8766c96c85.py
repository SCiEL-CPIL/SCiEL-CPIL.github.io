"""Optimized search for rank-minimal tensor decomposition of 2x4 * 4x4 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _verify_fast(U, V, W, n=2, m=4, p=4):
    """Faster verification using numpy."""
    n, m, p = 2, 4, 4
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
        
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    return np.allclose(T, np.ones((n,m,p)), atol=1e-9)

def _try_round(U, V, W, n=2, m=4, p=4):
    """Attempt to round to half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook rank 32
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    random.seed(42)
    np.random.seed(42)
    
    # Try to find lower rank using randomized search with verification
    # The key insight: we need T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Strategy: try ranks from 16 upward
    # Use random initialization and local search
    
    for target_rank in range(16, 32):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.uniform(-1.0, 1.0, (n*m, target_rank))
            V = np.random.uniform(-1.0, 1.0, (m*p, target_rank))
            W = np.random.uniform(-1.0, 1.0, (p*n, target_rank))
            
            # Local search: gradient-like updates
            # We want to minimize sum_{i,j,k} (T[i,j,k] - 1)^2
            # This is a non-convex problem, but gradient descent can find local minima
            
            lr = 0.05
            momentum_U = np.zeros_like(U)
            momentum_V = np.zeros_like(V)
            momentum_W = np.zeros_like(W)
            beta = 0.9
            
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient using vectorized operations
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # dL/dU[i*m+j,c] = sum_{i,j,k} (T[i,j,k]-1) * V[j*p+k,c] * W[k*n+i,c]
                # Note: U[i*m+j,c] only appears in T[i,j,k] for the specific i,j.
                # So dL/dU[i*m+j,c] = (T[i,j,k]-1) * V[j*p+k,c] * W[k*n+i,c] ?
                # No, U[i*m+j,c] appears in T[i', j', k] where i'=i, j'=j?
                # Yes, T[i,j,k] depends on U[i*m+j,c].
                # So dL/dU[i*m+j,c] = (T[i,j,k]-1) * V[j*p+k,c] * W[k*n+i,c]
                
                # Let's compute T first
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
                
                error = T - 1.0
                
                # dU[i*m+j, c] = error[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
                # We can compute this with einsum
                # error shape: (n, m, p)
                # V shape: (m, p, rank)
                # W shape: (p, n, rank)
                
                # dU[i, j, c] = sum_k error[i, j, k] * V[j, k, c] * W[k, i, c]
                dU = np.einsum('ijk,jkc,kic->ijc', error, V_r, W_r, optimize=True)
                
                # dV[j, k, c] = sum_i error[i, j, k] * U[i, j, c] * W[k, i, c]
                dV = np.einsum('ijk,ijc,kic->jkc', error, U_r, W_r, optimize=True)
                
                # dW[k, i, c] = sum_j error[i, j, k] * U[i, j, c] * V[j, k, c]
                dW = np.einsum('ijk,ijc,jkc->kic', error, U_r, V_r, optimize=True)
                
                # Reshape back
                dU = dU.reshape(n*m, target_rank)
                dV = dV.reshape(m*p, target_rank)
                dW = dW.reshape(p*n, target_rank)
                
                # Momentum update
                momentum_U = beta * momentum_U + (1 - beta) * dU
                momentum_V = beta * momentum_V + (1 - beta) * dV
                momentum_W = beta * momentum_W + (1 - beta) * dW
                
                U -= lr * momentum_U
                V -= lr * momentum_V
                W -= lr * momentum_W
                
                # Check convergence
                max_err = np.max(np.abs(T - 1.0))
                
                if max_err < 1e-8:
                    # Found a solution!
                    # Try to round
                    rounded = _try_round(U, V, W)
                    if rounded:
                        best_rank = target_rank
                        best_U = rounded[0].tolist()
                        best_V = rounded[1].tolist()
                        best_W = rounded[2].tolist()
                        return {"U": best_U, "V": best_V, "W": best_W}
                    else:
                        # Use unrounded but verify
                        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
                            best_rank = target_rank
                            best_U = U.tolist()
                            best_V = V.tolist()
                            best_W = W.tolist()
                            return {"U": best_U, "V": best_V, "W": best_W}
                    break
                elif max_err < 1e-4:
                    # If we get close, try a few more steps with smaller lr
                    lr *= 0.5
            
    # Return the best found (default is schoolbook)
    return {"U": best_U, "V": best_V, "W": best_W}
