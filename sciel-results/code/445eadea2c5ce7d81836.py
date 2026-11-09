"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safe margin for 120s limit
    
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    num_constraints = n * m * p  # 18
    
    def check_solution(U, V, W, rank):
        # Vectorized check
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = (U_sb, V_sb, W_sb, rank_sb)
    
    def local_search(U, V, W, rank, max_iter=500):
        """Vectorized gradient-based local search with L1 sparsity penalty."""
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Compute residuals
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            residuals = T - 1.0
            total_err = np.sum(residuals**2)
            
            if total_err < best_err:
                best_err = total_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-14:
                break
            
            # Compute gradients
            gU = np.einsum('ijk,jkc,kic->ijc', residuals, V_t, W_t)
            gV = np.einsum('ijk,ijc,kic->jkc', residuals, U_t, W_t)
            gW = np.einsum('ijk,ijc,jkc->kic', residuals, U_t, V_t)
            
            # L1 sparsity penalty gradient (sign function)
            # We want to drive some columns to zero
            sparsity_weight = 0.01 * (1.0 + it * 0.1)  # Increase sparsity pressure over time
            
            # Add L1 penalty gradient: sign(U) * sparsity_weight
            # But we only want to penalize the norm of each column, not individual elements
            # Simpler: add a small L2 penalty to encourage small norms
            # Actually, let's just use a small constant penalty on all elements
            # This encourages the optimizer to find solutions with smaller values
            
            # Adaptive learning rate
            lr = 0.05 / (1.0 + it * 0.01)
            
            # Update with gradient
            U_t -= lr * gU
            V_t -= lr * gV
            W_t -= lr * gW
            
            # Apply soft thresholding to encourage sparsity
            # This is a proximal step for L1 regularization
            threshold = 0.01 * (1.0 + it * 0.05)
            U_t[np.abs(U_t) < threshold] = 0
            V_t[np.abs(V_t) < threshold] = 0
            W_t[np.abs(W_t) < threshold] = 0
        
        return best_U, best_V, best_W, best_err
    
    # Try to find lower rank by starting from perturbed schoolbook
    for target_rank in [15, 16, 17]:
        if time.time() > deadline - 10:
            break
        
        found_better = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            U = U_sb.copy()
            V = V_sb.copy()
            W = W_sb.copy()
            
            # Add noise
            U += np.random.randn(nm, 18) * 0.1
            V += np.random.randn(mp, 18) * 0.1
            W += np.random.randn(pn, 18) * 0.1
            
            U, V, W, err = local_search(U, V, W, 18, max_iter=300)
            
            if err < 1e-10:
                # Try to prune columns
                norms_U = np.linalg.norm(U, axis=0)
                norms_V = np.linalg.norm(V, axis=0)
                norms_W = np.linalg.norm(W, axis=0)
                
                max_norms = np.max(np.vstack([norms_U, norms_V, norms_W]), axis=0)
                sorted_indices = np.argsort(max_norms)
                
                active_indices = sorted_indices[:target_rank]
                
                U_p = U[:, active_indices]
                V_p = V[:, active_indices]
                W_p = W[:, active_indices]
                
                if check_solution(U_p, V_p, W_p, target_rank):
                    if target_rank < best[3]:
                        best = (U_p, V_p, W_p, target_rank)
                    found_better = True
                    break
        
        if found_better:
            break
    
    U, V, W, rank = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
