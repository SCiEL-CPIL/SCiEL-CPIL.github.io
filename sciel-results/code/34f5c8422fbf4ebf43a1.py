import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target: 2 x 3 x 9
    # Strategy: Decompose into 3 blocks of 2 x 3 x 3
    # Find rank-minimal decomposition for 2 x 3 x 3
    # Then replicate 3 times
    
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Target tensor for 2x3x3
    # Indices: U (i*m+j), V (j*p+k), W (k*n+i)
    # i in 0..1, j in 0..2, k in 0..2
    # U idx: 3i+j
    # V idx: 3j+k
    # W idx: 2k+i
    
    size_u = n_sub * m_sub
    size_v = m_sub * p_sub
    size_w = p_sub * n_sub
    
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                u_idx = i * m_sub + j
                v_idx = j * p_sub + k
                w_idx = k * n_sub + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Try to find rank 15 decomposition
    best_rank = n_sub * m_sub * p_sub # 18
    best_U = None
    best_V = None
    best_W = None
    
    deadline = time.time() + 110 # Leave 10s margin
    
    for rank in [15, 16, 17, 18]:
        if time.time() > deadline:
            break
            
        # Random init
        np.random.seed(42)
        U = np.random.randn(size_u, rank)
        V = np.random.randn(size_v, rank)
        W = np.random.randn(size_w, rank)
        
        # Optimize
        # Flatten U, V, W
        def flatten_params(params):
            u = params[:size_u*rank].reshape(size_u, rank)
            v = params[size_u*rank : size_u*rank + size_v*rank].reshape(size_v, rank)
            w = params[size_u*rank + size_v*rank :].reshape(size_w, rank)
            return u, v, w
        
        def objective(params):
            u, v, w = flatten_params(params)
            # Compute T' = sum_r u_r v_r^T w_r^T ?
            # T'_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
            # This is an outer product sum.
            # We can compute this efficiently using einsum or loops.
            # T' = np.einsum('ar,br,cr->abc', u, v, w)
            # This might be slow for large rank, but here sizes are small.
            t_prime = np.einsum('ar,br,cr->abc', u, v, w)
            diff = t_prime - T
            return np.sum(diff**2)
            
        # Use scipy if available, else simple gradient descent
        try:
            from scipy.optimize import minimize
            # Initial params
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B or similar
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-12})
            
            if res.success or res.fun < 1e-8:
                u_opt, v_opt, w_opt = flatten_params(res.x)
                # Round to half-integers
                u_rounded = np.round(u_opt * 2) / 2
                v_rounded = np.round(v_opt * 2) / 2
                w_rounded = np.round(w_opt * 2) / 2
                
                # Verify exactness
                t_check = np.einsum('ar,br,cr->abc', u_rounded, v_rounded, w_rounded)
                if np.allclose(t_check, T, atol=1e-9):
                    best_rank = rank
                    best_U = u_rounded
                    best_V = v_rounded
                    best_W = w_rounded
                    break
        except Exception as e:
            pass

    # If no better found, use schoolbook for 2x3x3
    if best_U is None:
        best_rank = 18
        best_U = np.zeros((size_u, best_rank))
        best_V = np.zeros((size_v, best_rank))
        best_W = np.zeros((size_w, best_rank))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    best_U[i * m_sub + j, col] = 1
                    best_V[j * p_sub + k, col] = 1
                    best_W[k * n_sub + i, col] = 1
                    col += 1

    # Now construct 2x3x9 using 3 copies of the 2x3x3 decomposition
    # The 2x3x9 tensor is the direct sum of three 2x3x3 tensors
    # We need to map the indices correctly.
    # 2x3x9:
    # n=2, m=3, p=9
    # U: 2*3=6 rows, R cols
    # V: 3*9=27 rows, R cols
    # W: 9*2=18 rows, R cols
    
    total_rank = best_rank * 3
    U_final = np.zeros((6, total_rank))
    V_final = np.zeros((27, total_rank))
    W_final = np.zeros((18, total_rank))
    
    # For each block q in 0..2
    # The block corresponds to k indices 3q, 3q+1, 3q+2 in the 2x3x9 problem
    # In the 2x3x3 sub-problem, k is 0,1,2
    # So for block q:
    # U indices are the same (i,j)
    # V indices are (j, 3q+k_sub)
    # W indices are (3q+k_sub, i)
    
    for q in range(3):
        offset = q * best_rank
        # U is same for all blocks
        U_final[:, offset:offset+best_rank] = best_U
        
        # V: map (j, k_sub) to (j, 3q+k_sub)
        for j in range(3):
            for k_sub in range(3):
                v_sub_idx = j * 3 + k_sub
                v_full_idx = j * 9 + (3 * q + k_sub)
                V_final[v_full_idx, offset:offset+best_rank] = best_V[v_sub_idx, :]
                
        # W: map (k_sub, i) to (3q+k_sub, i)
        # W index in sub: k_sub * 2 + i
        # W index in full: (3q+k_sub) * 2 + i
        for k_sub in range(3):
            for i in range(2):
                w_sub_idx = k_sub * 2 + i
                w_full_idx = (3 * q + k_sub) * 2 + i
                W_final[w_full_idx, offset:offset+best_rank] = best_W[w_sub_idx, :]

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
