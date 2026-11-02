import numpy as np
import time
import sys
import scipy.optimize as opt

def check_decomposition_fast(U, V, W, n, m, p):
    """Verify the tensor decomposition identity using NumPy broadcasting."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Indices:
    # U index: a*m + b
    # V index: c*p + d
    # W index: k*n + i
    # Target: T[a, b, c, d, k, i] = 1 if a==i and b==c and d==k else 0
    
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    # Compute S[a, b, c, d, k, i] = sum_r U[a, b, r] V[c, d, r] W[k, i, r]
    S = np.einsum('abr,cdr,kir->abcdki', U_reshaped, V_reshaped, W_reshaped)
    
    # Construct Target T
    a_idx = np.arange(n)
    b_idx = np.arange(m)
    c_idx = np.arange(m)
    d_idx = np.arange(p)
    k_idx = np.arange(p)
    i_idx = np.arange(n)
    
    delta_a_i = (a_idx[:, None, None, None, None, None] == i_idx[None, None, None, None, None, :]).astype(float)
    delta_b_c = (b_idx[None, :, None, None, None, None] == c_idx[None, None, :, None, None, None]).astype(float)
    delta_d_k = (d_idx[None, None, None, :, None, None] == k_idx[None, None, None, None, :, None]).astype(float)
    
    T = delta_a_i * delta_b_c * delta_d_k
    
    return np.allclose(S, T, atol=1e-9)

def get_block_solution():
    n, m, p = 2, 3, 6
    rank_half = 17
    rank_total = 34
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    U_half = np.zeros((6, rank_half))
    V_half = np.zeros((9, rank_half))
    W_half = np.zeros((6, rank_half))
    
    # Strassen on A(0,1) x B(0,1) -> C(0,1)
    U_half[0, 0] = 1; U_half[4, 0] = 1
    V_half[0, 0] = 1; V_half[4, 0] = 1
    W_half[0, 0] = 1; W_half[3, 0] = 1

    U_half[3, 1] = 1; U_half[4, 1] = 1
    V_half[0, 1] = 1
    W_half[1, 1] = 1

    U_half[0, 2] = 1
    V_half[1, 2] = 1; V_half[4, 2] = -1
    W_half[2, 2] = 1; W_half[3, 2] = 1

    U_half[4, 3] = 1
    V_half[3, 3] = 1; V_half[0, 3] = -1
    W_half[0, 3] = 1; W_half[1, 3] = 1

    U_half[0, 4] = 1; U_half[1, 4] = 1
    V_half[4, 4] = 1
    W_half[0, 4] = -1; W_half[2, 4] = 1

    U_half[3, 5] = 1; U_half[0, 5] = -1
    V_half[0, 5] = 1; V_half[1, 5] = 1
    W_half[3, 5] = 1

    U_half[1, 6] = 1; U_half[4, 6] = -1
    V_half[3, 6] = 1; V_half[4, 6] = 1
    W_half[0, 6] = 1; W_half[3, 6] = 1

    # Column 2 of B: A(0,1,2) x B(0,1,2) col 2 -> C col 2
    U_half[0, 7] = 1
    V_half[2, 7] = 1
    W_half[4, 7] = 1

    U_half[1, 8] = 1
    V_half[5, 8] = 1
    W_half[4, 8] = 1

    U_half[3, 9] = 1
    V_half[2, 9] = 1
    W_half[5, 9] = 1

    U_half[4, 10] = 1
    V_half[5, 10] = 1
    W_half[5, 10] = 1

    # A col 2 x B(0,1,2) -> C(0,1,2)
    U_half[2, 11] = 1
    V_half[6, 11] = 1
    W_half[0, 11] = 1

    U_half[2, 12] = 1
    V_half[7, 12] = 1
    W_half[2, 12] = 1

    U_half[2, 13] = 1
    V_half[8, 13] = 1
    W_half[4, 13] = 1

    U_half[5, 14] = 1
    V_half[6, 14] = 1
    W_half[1, 14] = 1

    U_half[5, 15] = 1
    V_half[7, 15] = 1
    W_half[3, 15] = 1

    U_half[5, 16] = 1
    V_half[8, 16] = 1
    W_half[5, 16] = 1

    # Combine two 2x3x3 blocks
    U[:, :17] = U_half
    U[:, 17:] = U_half

    for j in range(3):
        for k in range(3):
            V[j * 6 + k, :17] = V_half[j * 3 + k, :17]
            V[j * 6 + 3 + k, 17:] = V_half[j * 3 + k, :17]

    for k in range(3):
        for i in range(2):
            W[k * 2 + i, :17] = W_half[k * 2 + i, :17]
            W[(k + 3) * 2 + i, 17:] = W_half[k * 2 + i, :17]

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    n, m, p = 2, 3, 6
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute Target Tensor T
    T_tensor = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                T_tensor[a, b, c, d, k, i] = 1.0
    T_flat = T_tensor.flatten()
    
    # Get the known rank 34 solution
    U34, V34, W34 = get_block_solution()
    rank34 = 34
    
    # Verify rank 34 solution
    if check_decomposition_fast(U34, V34, W34, n, m, p):
        best_U, best_V, best_W = U34, V34, W34
        best_rank = rank34
    else:
        # Fallback to trivial if 34 fails (should not happen)
        best_rank = n*m*p
        best_U = np.zeros((dim_u, best_rank))
        best_V = np.zeros((dim_v, best_rank))
        best_W = np.zeros((dim_w, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Strategy: Try to find rank 33 solution via NLS optimization
    # We will attempt to collapse the rank 34 solution into rank 33.
    # This is a non-convex problem. We use multiple random initializations by
    # zeroing out different terms from the rank 34 solution and adding small noise.
    
    target_rank = 33
    
    # Define residual function
    # x = [U.flatten(), V.flatten(), W.flatten()]
    def residual(x):
        U_r = x[:dim_u * target_rank].reshape(dim_u, target_rank)
        V_r = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
        W_r = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
        
        U_reshaped = U_r.reshape(n, m, -1)
        V_reshaped = V_r.reshape(m, p, -1)
        W_reshaped = W_r.reshape(p, n, -1)
        
        S = np.einsum('abr,cdr,kir->abcdki', U_reshaped, V_reshaped, W_reshaped)
        return (S - T_tensor).flatten()
    
    # We will try a few random "projections" of the rank 34 solution to rank 33.
    # Idea: Pick a random subset of 33 columns from the 34, or zero out one column and optimize.
    # Zeroing out one column and optimizing is effectively "re-fitting" the tensor with 33 terms.
    # Since the rank 34 solution is exact, removing one term creates a residual.
    # The optimizer will try to adjust the remaining 33 terms to absorb that residual.
    
    num_attempts = 20
    rng = np.random.default_rng(42)
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 20:
            break
            
        # Create initial guess for rank 33
        # Option A: Randomly drop one term from rank 34
        # Option B: Randomly perturb rank 33 terms (harder to initialize)
        
        # Let's try dropping different terms from the rank 34 solution.
        # The rank 34 solution consists of two blocks of 17.
        # Dropping a term from block 1 or block 2 is symmetric in structure, 
        # but the specific indices matter.
        
        drop_idx = attempt % rank34
        
        U_init = U34[:, :target_rank].copy()
        V_init = V34[:, :target_rank].copy()
        W_init = W34[:, :target_rank].copy()
        
        # If we drop term `drop_idx`, we need to map the remaining 34 columns to 33.
        # This is tricky because the "columns" are not inherently ordered in a way that 
        # allows simple truncation if the dropped one is in the middle.
        # However, for NLS, the order of terms doesn't matter, only the set of vectors.
        # So we can just take the first 33 columns? No, that assumes the last column is the one to drop.
        # Better: Create a mask.
        
        mask = np.ones(rank34, dtype=bool)
        mask[drop_idx] = False
        
        U_init = U34[:, mask]
        V_init = V34[:, mask]
        W_init = W34[:, mask]
        
        # Add small noise to escape local minima? 
        # Actually, starting from the exact solution minus one term is a good start.
        # But the "shape" of the residual might be hard for 33 terms to absorb if the rank is truly 34.
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Run optimization
        # Use 'trf' for bounded? No bounds needed really, but 'lm' is faster if unconstrained.
        # However, 'lm' requires m >= n. Here m=1296, n=33*36=1188. m > n, so 'lm' works.
        # 'lm' (Levenberg-Marquardt) is good for NLS.
        
        try:
            res = opt.least_squares(residual, x0, method='lm', max_nfev=1000)
            # Check if solution is valid
            if res.cost < 1e-10:
                U_sol = res.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                V_sol = res.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W_sol = res.x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                
                if check_decomposition_fast(U_sol, V_sol, W_sol, n, m, p):
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        except Exception as e:
            pass
            
        # Alternative: Random perturbation of a rank 33 subset
        # This is less structured but might find a different basin.
        if attempt % 5 == 0:
             # Pick random 33 columns from 34? No, just random noise init is too hard.
             # Let's try perturbing the dropped-column init.
             pass

    # If rank 33 failed, return rank 34
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
