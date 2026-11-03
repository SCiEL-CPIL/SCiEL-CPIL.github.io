"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Target tensor T
    # T has shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = i*m + j, v = j*p + k, w = k*n + i
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def error_func(params, rank):
        # params: flat array of size rank * (U_dim + V_dim + W_dim)
        U_flat, V_flat, W_flat = params[:rank*U_dim], params[rank*U_dim:2*rank*U_dim], params[2*rank*U_dim:]
        U = U_flat.reshape(rank, U_dim).T # (U_dim, rank)
        V = V_flat.reshape(rank, V_dim).T # (V_dim, rank)
        W = W_flat.reshape(rank, W_dim).T # (W_dim, rank)
        
        # Compute sum_r U[:,r] outer V[:,r] outer W[:,r]
        # This is expensive to do explicitly for large rank.
        # Instead, we can compute the residual tensor.
        # Res[u,v,w] = T[u,v,w] - sum_r U[u,r] V[v,r] W[w,r]
        
        # Efficient calculation:
        # For each r, add outer product.
        # Or use einsum.
        # C = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This might be slow for rank 40 and dims 6, 21, 14.
        # 6*21*14 = 1764 entries. Rank 40.
        # 40 * 1764 * 3 mults ~ 200k ops. Very fast.
        
        C = np.einsum('ur,vr,wr->uvw', U, V, W)
        residual = T - C
        return np.sum(residual**2)

    def get_matrices(params, rank):
        U_flat, V_flat, W_flat = params[:rank*U_dim], params[rank*U_dim:2*rank*U_dim], params[2*rank*U_dim:]
        U = U_flat.reshape(rank, U_dim).T
        V = V_flat.reshape(rank, V_dim).T
        W = W_flat.reshape(rank, W_dim).T
        return U, V, W

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')

    # Schoolbook init
    U_school = np.zeros((U_dim, best_rank))
    V_school = np.zeros((V_dim, best_rank))
    W_school = np.zeros((W_dim, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try ranks from lower bound up to schoolbook
    # Lower bound: n*m + m*p + p*n - 1 = 6 + 21 + 14 - 1 = 40
    for rank in range(40, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Initialize with schoolbook (truncate if rank < 42)
        if rank < best_rank:
            U_init = U_school[:, :rank]
            V_init = V_school[:, :rank]
            W_init = W_school[:, :rank]
        else:
            U_init = U_school.copy()
            V_init = V_school.copy()
            W_init = W_school.copy()
            
        # Add small random noise to break symmetry
        U_init = U_init + np.random.normal(0, 0.1, U_init.shape)
        V_init = V_init + np.random.normal(0, 0.1, V_init.shape)
        W_init = W_init + np.random.normal(0, 0.1, W_init.shape)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try a few random restarts for each rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            if attempt > 0:
                # Random init
                U_r = np.random.normal(0, 1, (U_dim, rank))
                V_r = np.random.normal(0, 1, (V_dim, rank))
                W_r = np.random.normal(0, 1, (W_dim, rank))
                params_init = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
                
            res = minimize(error_func, params_init, args=(rank,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                U, V, W = get_matrices(res.x, rank)
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                
            # If error is very small, we found a valid decomposition
            if best_err < 1e-10:
                # Check if we can go lower? No, we are iterating up.
                # But we want the MINIMAL rank.
                # So if we find rank 40, we return it.
                # If we find rank 41, we keep searching for 40? 
                # The loop goes 40, 41, 42.
                # If 40 fails, 41 might succeed.
                # If 41 succeeds, we return 41 unless 40 also succeeds later?
                # The loop is ascending. So if 40 succeeds, we break.
                break
                
        if best_rank == rank and best_err < 1e-10:
            break

    if best_U is None:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
