"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses ALS optimization to find a rank-44 decomposition, with rounding to half-integers.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    U_dim = n * m   # 8
    V_dim = m * p   # 24
    W_dim = p * n   # 12
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the target tensor
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook solution (rank 48)
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school))
    V_school = np.zeros((V_dim, rank_school))
    W_school = np.zeros((W_dim, rank_school))
    for idx in range(rank_school):
        i = idx // (m * p)
        rem = idx % (m * p)
        j = rem // p
        k = rem % p
        U_school[i * m + j, idx] = 1.0
        V_school[j * p + k, idx] = 1.0
        W_school[k * n + i, idx] = 1.0
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank solutions using ALS
    for target_rank in [44, 45, 46]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(30):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart * 13 + target_rank * 7)
            
            U_mat = np.random.randn(U_dim, target_rank) * 0.3
            V_mat = np.random.randn(V_dim, target_rank) * 0.3
            W_mat = np.random.randn(W_dim, target_rank) * 0.3
            
            for it in range(300):
                if time.time() > deadline - 15:
                    break
                
                # Update W
                A_mat = np.zeros((U_dim * V_dim * W_dim, target_rank * W_dim))
                b_vec = np.zeros(U_dim * V_dim * W_dim)
                eq_idx = 0
                for a in range(U_dim):
                    for b in range(V_dim):
                        for c in range(W_dim):
                            for r in range(target_rank):
                                A_mat[eq_idx, c * target_rank + r] = U_mat[a, r] * V_mat[b, r]
                            b_vec[eq_idx] = T[a, b, c]
                            eq_idx += 1
                
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_mat = W_flat.reshape(W_dim, target_rank)
                
                # Update U
                A_mat = np.zeros((V_dim * W_dim, target_rank))
                for r in range(target_rank):
                    A_mat[:, r] = np.outer(V_mat[:, r], W_mat[:, r]).ravel()
                
                U_mat = np.zeros((U_dim, target_rank))
                for a in range(U_dim):
                    U_mat[a, :], _, _, _ = np.linalg.lstsq(A_mat, T[a, :, :].ravel(), rcond=None)
                
                # Update V
                A_mat = np.zeros((U_dim * W_dim, target_rank))
                for r in range(target_rank):
                    A_mat[:, r] = np.outer(U_mat[:, r], W_mat[:, r]).ravel()
                
                V_mat = np.zeros((V_dim, target_rank))
                for b in range(V_dim):
                    V_mat[b, :], _, _, _ = np.linalg.lstsq(A_mat, T[:, b, :].ravel(), rcond=None)
            
            # Verify and try to round
            T_recon = np.zeros((U_dim, V_dim, W_dim))
            for r in range(target_rank):
                T_recon += np.outer(np.outer(U_mat[:, r], V_mat[:, r]), W_mat[:, r])
            
            err = np.max(np.abs(T_recon - T))
            if err < 1e-8:
                # Try rounding to half-integers
                for scale in [1, 2, 4]:
                    U_round = np.round(U_mat * scale) / scale
                    V_round = np.round(V_mat * scale) / scale
                    W_round = np.round(W_mat * scale) / scale
                    
                    T_recon_round = np.zeros((U_dim, V_dim, W_dim))
                    for r in range(target_rank):
                        T_recon_round += np.outer(np.outer(U_round[:, r], V_round[:, r]), W_round[:, r])
                    
                    err_round = np.max(np.abs(T_recon_round - T))
                    if err_round < 1e-9:
                        best_rank = target_rank
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        found = True
                        break
                
                if found:
                    break
        
        if found:
            break
    
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
