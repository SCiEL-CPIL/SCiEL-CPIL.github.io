"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_block(U, V, W, n, m, p, tol=1e-6):
    """
    Verify decomposition for n x m x p.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Check sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1
    # We can vectorize this slightly by computing for all k for a fixed i,j
    # Or just loop, as dimensions are small.
    for i in range(n):
        for j in range(m):
            # U[i,j,:] is a vector of length rank
            u_vec = U_t[i, j, :]
            # We need sum_k (sum_c u_vec[c] * V[j,k,c] * W[k,i,c])
            # Let's compute the vector S_k = sum_c u_vec[c] * V[j,k,c] * W[k,i,c] for all k
            # V[j, :, :] is (p, rank), W[:, i, :] is (p, rank)
            v_mat = V_t[j, :, :] # (p, rank)
            w_mat = W_t[:, i, :] # (p, rank)
            
            # element-wise product of u_vec (rank,) with v_mat (p, rank) -> (p, rank)
            # then multiply by w_mat (p, rank) -> (p, rank)
            # then sum over rank axis -> (p,)
            terms = (u_vec[None, :] * v_mat) * w_mat
            sums = np.sum(terms, axis=1)
            
            if not np.allclose(sums, 1.0, atol=tol):
                return False
    return True

def _optimize_1x5x5(rank, max_iter=500, tol=1e-8, seed=42, time_limit=10.0):
    """
    Find a rank-rank decomposition for 1x5x5 matrix multiplication.
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 1, 5, 5
    
    def objective(x):
        # U: 1*5*rank = 5r
        # V: 5*5*rank = 25r
        # W: 5*1*rank = 5r
        U_flat = x[:5*rank]
        V_flat = x[5*rank : 5*rank + 25*rank]
        W_flat = x[5*rank + 25*rank : 5*rank + 30*rank]
        
        U = U_flat.reshape(1, 5, rank)
        V = V_flat.reshape(5, 5, rank)
        W = W_flat.reshape(5, 1, rank)
        
        # Error: sum_c U[0,j,c] * V[j,k,c] * W[k,0,c] - 1
        err = 0.0
        for j in range(5):
            for k in range(5):
                s = np.sum(U[0, j, :] * V[j, k, :] * W[k, 0, :])
                err += (s - 1.0) ** 2
        return err

    def gradient(x):
        U_flat = x[:5*rank]
        V_flat = x[5*rank : 5*rank + 25*rank]
        W_flat = x[5*rank + 25*rank : 5*rank + 30*rank]
        
        U = U_flat.reshape(1, 5, rank)
        V = V_flat.reshape(5, 5, rank)
        W = W_flat.reshape(5, 1, rank)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for j in range(5):
            for k in range(5):
                s = np.sum(U[0, j, :] * V[j, k, :] * W[k, 0, :])
                ds = 2.0 * (s - 1.0)
                # dU[0,j,:] += ds * V[j,k,:] * W[k,0,:]
                # dV[j,k,:] += ds * U[0,j,:] * W[k,0,:]
                # dW[k,0,:] += ds * U[0,j,:] * V[j,k,:]
                prod_vw = V[j, k, :] * W[k, 0, :]
                prod_uw = U[0, j, :] * W[k, 0, :]
                prod_uv = U[0, j, :] * V[j, k, :]
                
                dU[0, j, :] += ds * prod_vw
                dV[j, k, :] += ds * prod_uw
                dW[k, 0, :] += ds * prod_uv
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize with small random values
    # Scaling is important. If values are too large, objective blows up.
    # If too small, gradient is small.
    # Schoolbook has 1s and 0s.
    # Let's try random values in [-1, 1]
    
    U_init = np.random.randn(1, 5, rank) * 0.5
    V_init = np.random.randn(5, 5, rank) * 0.5
    W_init = np.random.randn(5, 1, rank) * 0.5
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    except Exception:
        return None, None, None

    if time.time() - start_time > time_limit:
        return None, None, None

    if result.fun < tol:
        U_flat = result.x[:5*rank]
        V_flat = result.x[5*rank : 5*rank + 25*rank]
        W_flat = result.x[5*rank + 25*rank : 5*rank + 30*rank]
        
        U = U_flat.reshape(5, rank) # 1*5 -> 5
        V = V_flat.reshape(25, rank) # 5*5 -> 25
        W = W_flat.reshape(5, rank) # 5*1 -> 5
        
        # Verify
        if _verify_block(U, V, W, 1, 5, 5, tol=1e-6):
            return U, V, W
        else:
            return None, None, None
    else:
        return None, None, None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find a better decomposition using block structure
    # 3x5x5 is 3 blocks of 1x5x5.
    # If we find rank r for 1x5x5, total rank is 3r.
    
    # Try ranks from 20 up to 24 (schoolbook for 1x5x5 is 25, so 3*25=75)
    # If we find rank 24, total is 72.
    # If we find rank 20, total is 60.
    
    found_better = False
    
    for target_rank_1x5x5 in range(20, 25):
        if time.time() > deadline - 10:
            break
            
        total_rank = n * target_rank_1x5x5
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
            if time.time() > deadline - 10:
                break
                
            U1, V1, W1 = _optimize_1x5x5(target_rank_1x5x5, max_iter=300, tol=1e-8, seed=seed, time_limit=5.0)
            
            if U1 is not None:
                # Construct the full 3x5x5 decomposition
                # U_full: (15, 3r)
                # V_full: (25, 3r)
                # W_full: (15, 3r)
                
                U_full = np.zeros((n*m, total_rank))
                V_full = np.zeros((m*p, total_rank))
                W_full = np.zeros((p*n, total_rank))
                
                for i in range(n):
                    offset = i * target_rank_1x5x5
                    # U rows: i*m .. i*m+m-1
                    # U1 rows: 0 .. m-1 (since 1x5x5, U is 5xrank)
                    U_full[i*m : i*m+m, offset : offset+target_rank_1x5x5] = U1
                    
                    # V rows: 0 .. m*p-1
                    # V1 rows: 0 .. m*p-1
                    V_full[:, offset : offset+target_rank_1x5x5] = V1
                    
                    # W rows: k*n .. k*n+n-1 for each k
                    # W_full row index: k*n + i
                    # W1 row index: k (since p x n = 5 x 1, W is 5xrank)
                    for k in range(p):
                        W_full[k*n + i, offset : offset+target_rank_1x5x5] = W1[k, :]
                
                # Verify full decomposition
                if _verify_block(U_full, V_full, W_full, n, m, p, tol=1e-6):
                    best_rank = total_rank
                    best_result = {
                        "U": U_full.tolist(),
                        "V": V_full.tolist(),
                        "W": W_full.tolist()
                    }
                    found_better = True
                    break # Found valid decomposition for this rank, try lower rank
        
        if found_better:
            # We found a decomposition at this rank. 
            # Since we are iterating from low to high, we should continue to try lower ranks 
            # if we haven't hit the theoretical minimum or if we want to ensure we found the *lowest*.
            # However, the loop `for target_rank_1x5x5 in range(20, 25)` goes 20, 21, 22, 23, 24.
            # If we succeed at 20, we break the inner loop, then the outer loop continues to 21? 
            # No, we want the MINIMAL rank. So if we find rank 20, that's better than 21.
            # But wait, the loop starts at 20. If we find 20, that's the best possible in this range.
            # So we should break the outer loop too if we find the lowest target we are trying.
            # Actually, the loop goes 20, 21, 22, 23, 24.
            # If we find a solution at 20, we should stop, because 20 is the lowest we are trying.
            # If we fail at 20, we try 21.
            # So the logic `if found_better: break` inside the seed loop is correct for the seed loop.
            # But we need to handle the outer loop.
            # If we found a solution at rank R, we should update best_rank.
            # And since we are going from low to high, if we find one at 20, we are done (for this range).
            # If we find one at 21, we might still find one at 20? No, we already tried 20 and failed.
            # So if we find a solution, we can break the outer loop as well? 
            # Yes, because we are iterating from low to high. The first success is the best (lowest) rank in this sequence.
            break

    return best_result
# EVOLVE-BLOCK-END
