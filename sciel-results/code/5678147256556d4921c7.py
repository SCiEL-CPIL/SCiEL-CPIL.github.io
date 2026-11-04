import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I_idx = i*m + j
                J_idx = j*p + k
                K_idx = k*n + i
                T[I_idx, J_idx, K_idx] = 1.0
                
    # Precompute index arrays for flattening
    # For U update:
    # A shape: (m*p * p*n, R)
    # A_flat[row, r] = V[J, r] * W[K, r]
    # row corresponds to (J, K)
    
    # We can precompute the structure of A, B, C updates
    # A is formed from V and W
    # B is formed from U and W
    # C is formed from U and V
    
    def try_rank(R, time_limit, seed_val=42):
        np.random.seed(seed_val)
        
        # Initialize
        # Use random initialization
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Precompute index mappings for b vectors
        # b_U[i] is the flattened T[i, :, :]
        b_U = [T[i, :, :].flatten() for i in range(n*m)]
        b_V = [T[:, j, :].flatten() for j in range(m*p)]
        b_W = [T[:, :, k].flatten() for k in range(p*n)]
        
        # Precompute index grids for constructing A, B, C
        # A rows: J in 0..m*p-1, K in 0..p*n-1
        Js = np.arange(m*p)
        Ks = np.arange(p*n)
        J_grid, K_grid = np.meshgrid(Js, Ks, indexing='ij')
        J_flat = J_grid.flatten()
        K_flat = K_grid.flatten()
        
        # For B: I in 0..n*m-1, K in 0..p*n-1
        Is = np.arange(n*m)
        I_grid, K_grid2 = np.meshgrid(Is, Ks, indexing='ij')
        I_flat = I_grid.flatten()
        K_flat2 = K_grid2.flatten()
        
        # For C: I in 0..n*m-1, J in 0..m*p-1
        I_grid2, J_grid2 = np.meshgrid(Is, Js, indexing='ij')
        I_flat2 = I_grid2.flatten()
        J_flat2 = J_grid2.flatten()
        
        start_time = time.time()
        
        for it in range(200):
            if time.time() > time_limit:
                return None
            
            # Update U
            # A[row, r] = V[J_flat[row], r] * W[K_flat[row], r]
            # V[J_flat, :] has shape (num_rows, R)
            # W[K_flat, :] has shape (num_rows, R)
            V_flat = V[J_flat, :] # (num_rows, R)
            W_flat = W[K_flat, :] # (num_rows, R)
            A = V_flat * W_flat   # (num_rows, R)
            
            for i in range(n*m):
                U[i, :], _, _, _ = np.linalg.lstsq(A, b_U[i], rcond=None)
                
            # Update V
            U_flat = U[I_flat, :]
            W_flat2 = W[K_flat2, :]
            B = U_flat * W_flat2
            for j in range(m*p):
                V[j, :], _, _, _ = np.linalg.lstsq(B, b_V[j], rcond=None)
                
            # Update W
            U_flat2 = U[I_flat2, :]
            V_flat2 = V[J_flat2, :]
            C = U_flat2 * V_flat2
            for k in range(p*n):
                W[k, :], _, _, _ = np.linalg.lstsq(C, b_W[k], rcond=None)
                
            # Check error
            # T_approx = np.einsum('ir, jr, kr -> ijk', U, V, W)
            # err = np.linalg.norm(T_approx - T)
            
            # Faster error check:
            # err = sum_r (U^T T V) ... no
            # Just compute norm of (U V W - T)
            # We can check residual in the linear systems?
            # Better to just compute the tensor error every few iterations
            if it % 5 == 0:
                T_approx = np.einsum('ir, jr, kr -> ijk', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < 1e-8:
                    # Try to snap to half-integers
                    U_int = U * 2
                    V_int = V * 2
                    W_int = W * 2
                    
                    U_round = np.round(U_int)
                    V_round = np.round(V_int)
                    W_round = np.round(W_int)
                    
                    U_check = U_round / 2.0
                    V_check = V_round / 2.0
                    W_check = W_round / 2.0
                    
                    T_check = np.einsum('ir, jr, kr -> ijk', U_check, V_check, W_check)
                    if np.linalg.norm(T_check - T) < 1e-9:
                        return U_check, V_check, W_check
                        
        return None

    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from 47 down to 40
    # If we don't find a lower rank, we return schoolbook (48)
    best_result = None
    
    # First, ensure we have a valid schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    for R in range(47, 39, -1):
        if time.time() > deadline:
            break
        # Try multiple seeds for this rank
        found = None
        for seed in range(10):
            if time.time() > deadline:
                break
            res = try_rank(R, deadline, seed_val=R * 100 + seed)
            if res is not None:
                found = res
                break
        if found is not None:
            U, V, W = found
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    return fallback
# EVOLVE-BLOCK-END
