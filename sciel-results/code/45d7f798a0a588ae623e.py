"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook decomposition
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Try to reduce rank via iterative projection/elimination
    # Strategy: try removing one factor at a time and see if we can re-approximate
    # with the remaining factors plus a small perturbation
    
    def residual_norm(U, V, W, T):
        """Compute ||T - sum_c U_c ⊗ V_c ⊗ W_c||_F"""
        # Efficient computation: reshape and use einsum
        # T_approx[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # We can compute this as: for each c, outer product
        # More efficient: contract step by step
        # First: M[i,j] = sum_c U[i,c]*V[j,c]  -> (n*m, m*p)
        # Then: T_approx[i,j,k] = sum_l M[i,l] * ... hmm, this doesn't factor nicely
        
        # Let's use: T_approx = sum_c np.einsum('ic,jc,kc->ijk', U[:,c], V[:,c], W[:,c])
        # But that's slow. Better: 
        # T_approx[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # = (U * V^T * W) contracted... 
        # Actually: let's compute as matrix multiplication chain
        # For fixed k: T_approx[:,:,k] = sum_c (U*W[k,c]) @ V[:,c].T
        # Or: T_approx = sum_c U[:,c,None,None] * V[None,:,c,None] * W[None,None,:,c]
        
        # Most efficient: use np.einsum
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
        return np.linalg.norm(T - T_approx)
    
    def try_remove_factor(U, V, W, T, c, max_iter=50, tol=1e-8):
        """Try to remove factor c by adjusting the other factors."""
        if time.time() > deadline:
            return None, None, None, False
        
        rank = U.shape[1]
        if rank <= 1:
            return None, None, None, False
        
        # Remove column c
        mask = [i for i in range(rank) if i != c]
        U_rem = U[:, mask]
        V_rem = V[:, mask]
        W_rem = W[:, mask]
        
        # We want to find U', V', W' of shape (n*m, rank-1), etc. such that
        # T ≈ sum_{c' != c} U'[c'] ⊗ V'[c'] ⊗ W'[c']
        # This is underdetermined in general, but we can try a simple approach:
        # Project the "missing" component onto the span of remaining factors
        
        # Compute the residual that factor c was providing
        # T_c = U[:,c,None,None] * V[None,:,c,None] * W[None,None,:,c]
        T_c = np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
        # Try to express T_c as a linear combination of the remaining rank-1 tensors
        # This is hard in general. Instead, let's use a simpler heuristic:
        # Check if factor c is "redundant" - i.e., if removing it doesn't change T much
        # relative to the total
        
        # Alternative approach: use random restarts with lower rank
        return None, None, None, False
    
    # Better approach: random initialization with lower rank, then optimize
    def random_init(rank_r, seed):
        rng = np.random.RandomState(seed)
        U_r = rng.randn(n*m, rank_r)
        V_r = rng.randn(m*p, rank_r)
        W_r = rng.randn(p*n, rank_r)
        # Normalize to avoid scale issues
        for c in range(rank_r):
            norm = np.linalg.norm(U_r[:,c]) * np.linalg.norm(V_r[:,c]) * np.linalg.norm(W_r[:,c])
            if norm > 1e-12:
                scale = norm ** (1.0/3.0)
                U_r[:,c] /= scale
                V_r[:,c] /= scale
                W_r[:,c] /= scale
        return U_r, V_r, W_r
    
    def optimize(U, V, W, T, max_iter=200, lr=0.01, tol=1e-10):
        """Simple gradient descent on ||T - sum U_c ⊗ V_c ⊗ W_c||^2"""
        if time.time() > deadline:
            return U, V, W
        
        rank_r = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_loss = residual_norm(U, V, W, T) ** 2
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # loss = ||T - sum_c U_c ⊗ V_c ⊗ W_c||^2
            # d_loss/dU[i,c] = -2 * sum_{j,k} T[i,j,k] * V[j,c] * W[k,c] + 2 * sum_{j,k} (sum_{c'} U[i,c']*V[j,c']*W[k,c']) * V[j,c] * W[k,c]
            
            # Compute T_approx
            T_approx = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
            diff = T_approx - T  # shape (n*m, m*p, p*n)
            
            # Gradient w.r.t. U: dU[i,c] = sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c]
            grad_U = np.einsum('ijk,jc,kc->ic', diff, V, W, optimize=True)
            grad_V = np.einsum('ijk,ic,kc->jc', diff, U, W, optimize=True)
            grad_W = np.einsum('ijk,ic,jc->kc', diff, U, V, optimize=True)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            loss = residual_norm(U, V, W, T) ** 2
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                break
            
            # Adaptive learning rate
            if it % 20 == 0 and it > 0:
                lr *= 0.9
        
        return best_U, best_V, best_W
    
    # Try various ranks
    best_result = (U.copy(), V.copy(), W.copy())
    best_rank = rank
    best_loss = 0.0
    
    # Try ranks from current down to some minimum
    # Known lower bound for 4x4 * 4x6: at least 4*4*6 / something
    # Strassen-like bounds suggest we might do better than 96
    min_rank_to_try = 40
    
    for r in range(rank - 1, min_rank_to_try - 1, -2):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_r, V_r, W_r = random_init(r, seed * 1000 + r)
            U_opt, V_opt, W_opt = optimize(U_r, V_r, W_r, T, max_iter=100, lr=0.05)
            
            loss = residual_norm(U_opt, V_opt, W_opt, T)
            if loss < 1e-6:
                # Found a valid decomposition with rank r
                best_result = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                best_rank = r
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # Convert to list format
    U_final, V_final, W_final = best_result
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
