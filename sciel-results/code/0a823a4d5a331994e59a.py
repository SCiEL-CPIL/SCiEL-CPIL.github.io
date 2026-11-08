"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 23 first, then 24, 25, 26
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
            
        result = _try_rank(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    # Fallback: return the rank-26 construction (known valid)
    return _rank_26_seed()

def _try_rank(n, m, p, target_rank, deadline):
    """
    Attempt to find a rank-`target_rank` decomposition using gradient-based optimization.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = target_rank
    
    # Build the target tensor T[i, k, j] = 1 if this term contributes to C[i,j]
    # T has shape (n, m, p) where T[i,k,j] = 1
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i, k, j] = 1.0
    
    # We want U (nm x rank), V (mp x rank), W (pn x rank)
    # such that for all i,k,j: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[i*p+j, r]
    
    np.random.seed(42 + target_rank)
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Multiple random restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 5:  # Leave 5s for finalization
            break
        
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        patience = 0
        max_patience = 50
        
        for step in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Compute current decomposition
            # Reshape for efficient computation
            U_3d = U.reshape(n, m, rank)  # U[i*m+k, r] -> U_3d[i, k, r]
            V_3d = V.reshape(m, p, rank)  # V[k*p+j, r] -> V_3d[k, j, r]
            W_3d = W.reshape(p, n, rank)  # W[i*p+j, r] -> W_3d[j, i, r]... wait
            
            # Actually, let me be more careful with indices
            # U row index = i*m + k, so U[i*m+k, r]
            # V row index = k*p + j, so V[k*p+j, r]  
            # W row index = i*p + j, so W[i*p+j, r]
            
            # For each (i,k,j), the contribution is sum_r U[i*m+k,r] * V[k*p+j,r] * W[i*p+j,r]
            
            # Compute the reconstructed tensor
            # recon[i,k,j] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[i*p+j,r]
            
            # This is expensive to compute naively. Let's use einsum
            # U_flat[i*m+k, r], V_flat[k*p+j, r], W_flat[i*p+j, r]
            
            # Create index arrays
            i_idx = np.arange(n).reshape(n, 1, 1)
            k_idx = np.arange(m).reshape(1, m, 1)
            j_idx = np.arange(p).reshape(1, 1, p)
            
            U_idx = (i_idx * m + k_idx).astype(int)  # shape (n, m, 1)
            V_idx = (k_idx * p + j_idx).astype(int)  # shape (1, m, p)
            W_idx = (i_idx * p + j_idx).astype(int)  # shape (n, 1, p)
            
            # Broadcast to (n, m, p)
            U_idx = np.broadcast_to(U_idx, (n, m, p))
            V_idx = np.broadcast_to(V_idx, (n, m, p))
            W_idx = np.broadcast_to(W_idx, (n, m, p))
            
            # Get the values
            U_vals = U[U_idx, :]  # shape (n, m, p, rank)
            V_vals = V[V_idx, :]  # shape (n, m, p, rank)
            W_vals = W[W_idx, :]  # shape (n, m, p, rank)
            
            # Compute reconstruction: sum over rank dimension
            recon = np.sum(U_vals * V_vals * W_vals, axis=-1)  # shape (n, m, p)
            
            # Compute loss
            loss = np.sum((recon - T) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                break
            
            # Compute gradients
            # dL/dU[i*m+k, r] = 2 * sum_{i',k',j'} (recon[i',k',j'] - T[i',k',j']) * V[k'*p+j', r] * W[i'*p+j', r] * delta(i*m+k == i'*m+k')
            # This is complex. Let's use a simpler approach: finite differences or analytic grad
            
            # Actually, let's use a simpler local search: random perturbations
            if step % 10 == 0:
                # Try random perturbations
                improved = False
                for _ in range(10):
                    U_pert = U + np.random.randn(nm, rank) * lr
                    V_pert = V + np.random.randn(mp, rank) * lr
                    W_pert = W + np.random.randn(pn, rank) * lr
                    
                    # Compute loss for perturbed
                    U_vals_p = U_pert[U_idx, :]
                    V_vals_p = V_pert[V_idx, :]
                    W_vals_p = W_pert[W_idx, :]
                    recon_p = np.sum(U_vals_p * V_vals_p * W_vals_p, axis=-1)
                    loss_p = np.sum((recon_p - T) ** 2)
                    
                    if loss_p < loss:
                        U, V, W = U_pert, V_pert, W_pert
                        loss = loss_p
                        improved = True
                        break
            
            # Adaptive learning rate
            if improved:
                lr = min(lr * 1.1, 0.1)
            else:
                lr = max(lr * 0.9, 0.001)
        
        # Check if we found a valid decomposition
        if best_loss < 1e-6:
            # Verify the solution
            U_check = best_U
            V_check = best_V
            W_check = best_W
            
            U_vals_c = U_check[U_idx, :]
            V_vals_c = V_check[V_idx, :]
            W_vals_c = W_check[W_idx, :]
            recon_c = np.sum(U_vals_c * V_vals_c * W_vals_c, axis=-1)
            
            if np.allclose(recon_c, T, atol=1e-5):
                # Convert to list format
                return {
                    "U": U_check.tolist(),
                    "V": V_check.tolist(),
                    "W": W_check.tolist()
                }
    
    # If we didn't find a valid decomposition for this rank, return None
    return None

def _rank_26_seed():
    """Replicate the seed's rank-26 construction."""
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T (Top-right 2x1 * Bottom-left 1x2) -> Rank 4
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # 3. A2 b (Top-left 2x2 * Top-right 2x1) -> Rank 4
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    
    # 4. a b3 (Top-right 2x1 * Bottom-right 1x1) -> Rank 2
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # 5. a^T B2 (Bottom-left 1x2 * Top-left 2x2) -> Rank 4
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    
    # 6. a3 b^T (Bottom-right 1x1 * Bottom-left 1x2) -> Rank 2
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # 7. a^T b (Bottom-left 1x2 * Top-right 2x1) -> Rank 2
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    
    # 8. a3 b3 (Bottom-right 1x1 * Bottom-right 1x1) -> Rank 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
