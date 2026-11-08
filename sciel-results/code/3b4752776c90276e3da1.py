"""Matrix multiplication 2x3x5 tensor decomposition - rank minimization."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] as a 3D array."""
    # T has shape (n*m, m*p, n*p)
    # T[(i,j), (j2,k), (i2,k2)] = 1 if j==j2 and i==i2 and k==k2, else 0
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T."""
    rank = U.shape[1]
    T_recon = np.zeros((n*m, m*p, n*p))
    for c in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    T_true = _build_tensor(n, m, p)
    return np.max(np.abs(T_recon - T_true)) < tol

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= n*m + m*p + n*p - n - m - p + 1 = 6+15+10-2-3-5+1 = 22
    # Try ranks from 22 upward
    
    T = _build_tensor(n, m, p)
    T_flat = T.reshape(-1)
    N = T_flat.shape[0]
    
    def _try_rank(rank, max_iters=2000, n_restarts=5):
        nonlocal best_result, best_rank
        for restart in range(n_restarts):
            if time.time() > deadline:
                return
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Gradient descent on F = 0.5 * ||sum_c u_c ⊗ v_c ⊗ w_c - T||^2
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline:
                    return
                # Compute residual
                recon = np.zeros((n*m, m*p, n*p))
                for c in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                residual = recon - T
                err = np.sum(residual**2)
                
                if err < 1e-12:
                    # Found exact decomposition
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    return True
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for c in range(rank):
                    dU[:,c] = np.einsum('abc,bc->a', residual, np.einsum('b,c->bc', V[:,c], W[:,c]))
                    dV[:,c] = np.einsum('abc,ac->b', residual, np.einsum('a,c->ac', U[:,c], W[:,c]))
                    dW[:,c] = np.einsum('abc,ab->c', residual, np.einsum('a,b->ab', U[:,c], V[:,c]))
                
                # Update with adaptive learning rate
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Scale invariance fix: normalize columns
                for c in range(rank):
                    norm_u = np.linalg.norm(U[:,c])
                    norm_v = np.linalg.norm(V[:,c])
                    norm_w = np.linalg.norm(W[:,c])
                    if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                        # Keep product of norms constant
                        target = (norm_u * norm_v * norm_w) ** (1/3)
                        U[:,c] *= target / norm_u
                        V[:,c] *= target / norm_v
                        W[:,c] *= target / norm_w
                
                if it % 50 == 0 and err > 1.0:
                    lr *= 0.9
                elif it % 50 == 0 and err < 0.1:
                    lr = min(lr * 1.1, 0.05)
        
        return False
    
    # Try ranks from lower bound upward
    lower_bound = n*m + m*p + n*p - n - m - p + 1  # = 22
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        success = _try_rank(rank, max_iters=1000, n_restarts=3)
        if success:
            break
    
    # Convert to lists
    return best_result

# EVOLVE-BLOCK-END
