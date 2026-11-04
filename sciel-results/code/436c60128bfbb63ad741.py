"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook (rank 84)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def validate_fast(U, V, W):
        """Vectorized check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # For each rank index c, compute the outer product pattern
        # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Sum over c should equal 1
        
        total = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            u_col = U_arr[:, c]
            v_col = V_arr[:, c]
            w_col = W_arr[:, c]
            
            # Build U tensor: U_t[i,j] = u_col[i*m+j]
            U_t = u_col.reshape(n, m)
            # Build V tensor: V_t[j,k] = v_col[j*p+k]
            V_t = v_col.reshape(m, p)
            # Build W tensor: W_t[k,i] = w_col[k*n+i]
            W_t = w_col.reshape(p, n)
            
            # T_c[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i]
            # Use einsum: U_t[i,j] * V_t[j,k] * W_t[k,i]
            # = (U_t[i,j] * W_t[k,i]) then * V_t[j,k]
            # Or: for each i,j: (U_t[i,j] * V_t[j,k] * W_t[k,i]).sum(k) should work
            # Actually we need the full tensor, not summed
            # T_c[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i]
            # This is an outer-like product with index coupling
            # Let's use broadcasting:
            # U_t[:, :, None] * V_t[None, :, :] * W_t.T[None, None, :]
            # W_t is (p,n), W_t.T is (n,p), W_t.T[k,i] -> we need W_t[k,i]
            # W_t[k*n+i] = w_col[k*n+i], so W_t as (p,n) matrix has W_t[k,i] = w_col[k*n+i]
            # So W_t is already in the right shape (p,n)
            # T_c[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i]
            # = U_t[i,j,1] * V_t[1,j,k] * W_t[k,1,i]... 
            # Let's use np.einsum
            # T_c = np.einsum('ij,jk,ki->ijk', U_t, V_t, W_t)
            
            T_c = np.einsum('ij,jk,ki->ijk', U_t, V_t, W_t)
            total += T_c
        
        if np.max(np.abs(total - 1.0)) > 1e-8:
            return False, np.max(np.abs(total - 1.0))
        return True, 0.0
    
    def try_rank(r, max_iters=30, seed=0):
        """Try to find a rank-r decomposition using random init + coordinate descent style optimization."""
        if r >= best_rank:
            return None
        rng = np.random.RandomState(seed)
        
        for attempt in range(max_iters):
            # Random initialization
            U = rng.randn(n * m, r) * 0.3
            V = rng.randn(m * p, r) * 0.3
            W = rng.randn(p * n, r) * 0.3
            
            # Coordinate descent: optimize each rank column
            for it in range(300):
                if time.time() > deadline:
                    return None
                
                # Pick a random rank index
                c = rng.randint(r)
                
                # For column c, we want to minimize sum_{i,j,k} (sum_{c'} U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1)^2
                # Fix all other columns, optimize U[:,c], V[:,c], W[:,c]
                
                # Compute contribution of all other columns
                residual = np.ones((n, m, p), dtype=float)
                for c2 in range(r):
                    if c2 == c:
                        continue
                    u_col = U[:, c2]
                    v_col = V[:, c2]
                    w_col = W[:, c2]
                    U_t = u_col.reshape(n, m)
                    V_t = v_col.reshape(m, p)
                    W_t = w_col.reshape(p, n)
                    T_c2 = np.einsum('ij,jk,ki->ijk', U_t, V_t, W_t)
                    residual -= T_c2
                
                # Now we need U[:,c], V[:,c], W[:,c] such that
                # U_t[i,j] * V_t[j,k] * W_t[k,i] ≈ residual[i,j,k]
                # This is a rank-1 tensor approximation problem
                # Use ALS or simple gradient descent on the log-scale
                
                # Simple approach: initialize with residual-based estimates
                # and do a few gradient steps
                u_c = U[:, c].copy()
                v_c = V[:, c].copy()
                w_c = W[:, c].copy()
                
                U_t = u_c.reshape(n, m)
                V_t = v_c.reshape(m, p)
                W_t = w_c.reshape(p, n)
                
                lr = 0.05
                for gd in range(50):
                    if time.time() > deadline:
                        return None
                    
                    T_c = np.einsum('ij,jk,ki->ijk', U_t, V_t, W_t)
                    diff = T_c - residual
                    
                    # Gradient w.r.t. U_t[i,j]: sum_k diff[i,j,k] * V_t[j,k] * W_t[k,i]
                    grad_U = np.einsum('ijk,jk,ki->ij', diff, V_t, W_t)
                    # Gradient w.r.t. V_t[j,k]: sum_i diff[i,j,k] * U_t[i,j] * W_t[k,i]
                    grad_V = np.einsum('ijk,ij,ki->jk', diff, U_t, W_t)
                    # Gradient w.r.t. W_t[k,i]: sum_j diff[i,j,k] * U_t[i,j] * V_t[j,k]
                    grad_W = np.einsum('ijk,ij,jk->ki', diff, U_t, V_t)
                    
                    U_t -= lr * grad_U
                    V_t -= lr * grad_V
                    W_t -= lr * grad_W
                    
                    # Renormalize occasionally to avoid scale drift
                    if gd % 10 == 9:
                        for mat, orig in [(U_t, u_c), (V_t, v_c), (W_t, w_c)]:
                            s = np.max(np.abs(mat))
                            if s > 0:
                                mat /= s
                                # Scale the other two compensating? No, just clamp
                            
                    if np.max(np.abs(diff)) < 1e-10:
                        break
                
                U[:, c] = U_t.flatten()
                V[:, c] = V_t.flatten()
                W[:, c] = W_t.flatten()
                
                # Check if valid
                valid, err = validate_fast(U, V, W)
                if valid:
                    return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Try various ranks from low to high
    # Lower bound: rank >= max(n*m, m*p, p*n) is not quite right
    # For matmul n×m × m×p, the tensor has dimension n*m*p
    # Rank lower bound: at least max(n*m, m*p, p*n) for the structure
    # Actually the trivial lower bound is max(n*m, m*p, p*n) but that's not tight
    # Let's try from 30 upward
    
    for r in range(30, min(best_rank + 1, 60)):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=10, seed=r * 13)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # If we haven't improved, try a few more random attempts at slightly higher ranks
    if best_rank == n * m * p:
        for r in range(40, min(60, n * m * p)):
            if time.time() > deadline:
                break
            result = try_rank(r, max_iters=15, seed=r * 7 + 100)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
