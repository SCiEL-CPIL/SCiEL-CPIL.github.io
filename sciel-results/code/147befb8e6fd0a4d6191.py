import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_U = n * m  # 9
    dim_V = m * p  # 15
    dim_W = p * n  # 15
    
    # Build target tensor T of shape (dim_W, dim_V, dim_U) or flattened
    # T[k, j, i] = 1 if indices match
    # i: A index (r_A * m + c_A)
    # j: B index (c_A * p + c_B)
    # k: C index (r_A * n + c_B)
    
    # We will use ALS to find U, V, W
    # U: (dim_U, rank)
    # V: (dim_V, rank)
    # W: (dim_W, rank)
    
    # Target tensor as a vector of length dim_W * dim_V * dim_U
    # Index mapping: idx = k * dim_V * dim_U + j * dim_U + i
    T_vec = np.zeros(dim_W * dim_V * dim_U)
    
    for r_A in range(n):
        for c_A in range(m):
            for c_B in range(p):
                i = r_A * m + c_A
                j = c_A * p + c_B
                k = r_A * n + c_B
                idx = k * dim_V * dim_U + j * dim_U + i
                T_vec[idx] = 1.0

    # Try ranks from 34 down to 25?
    # If none work, return best found or schoolbook.
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p
    best_residual = np.inf
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # We will try ranks 34, 35, 36
    # Start with 34 (better score)
    for rank in [34, 35, 36]:
        if time.time() > deadline:
            break
            
        # Try multiple restarts for each rank
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize
            # Scale initialization matters
            U = np.random.randn(dim_U, rank) * 0.1
            V = np.random.randn(dim_V, rank) * 0.1
            W = np.random.randn(dim_W, rank) * 0.1
            
            # ALS Iterations
            # We need to solve for U given V, W
            # Y = Z U^T
            # Y is (dim_W * dim_V) x dim_U
            # Z is (dim_W * dim_V) x rank
            
            # Precompute Z for current V, W
            # Z_{k, j, c} = V_{j, c} W_{k, c}
            
            residual = np.inf
            for it in range(1000):
                if time.time() > deadline:
                    break
                
                # Update U
                # Z is (dim_W * dim_V, rank)
                Z = np.zeros((dim_W * dim_V, rank))
                # Vectorized calculation
                # V is (dim_V, rank), W is (dim_W, rank)
                # Z[k, j, c] = V[j, c] * W[k, c]
                # We can use broadcasting
                # V.T is (rank, dim_V)
                # W.T is (rank, dim_W)
                # Outer product of W and V?
                # Z_{k, j, c} = W[k, c] * V[j, c]
                # Reshape to (dim_W, dim_V, rank)
                Z_3d = W[:, np.newaxis, :] * V[np.newaxis, :, :]
                Z = Z_3d.reshape(dim_W * dim_V, rank)
                
                # Y is (dim_W * dim_V, dim_U)
                Y = T_vec.reshape(dim_W * dim_V, dim_U)
                
                # Solve Z U^T = Y
                # U^T = (Z^T Z)^{-1} Z^T Y
                # Use least squares
                try:
                    U_new = np.linalg.lstsq(Z, Y, rcond=None)[0].T
                except:
                    break
                
                # Update V
                # Now solve for V
                # T_{k, j, i} = sum_c U_{i, c} V_{j, c} W_{k, c}
                # For fixed k, i:
                # T_{k, j, i} = sum_c V_{j, c} (U_{i, c} W_{k, c})
                # Y_V is (dim_W * dim_U, dim_V)
                # Z_V is (dim_W * dim_U, rank)
                # Z_V[k, i, c] = U[i, c] * W[k, c]
                Z_V_3d = U[np.newaxis, :, :] * W[:, np.newaxis, :]
                Z_V = Z_V_3d.reshape(dim_W * dim_U, rank)
                Y_V = T_vec.reshape(dim_W, dim_U, dim_V).transpose(0, 1, 2).reshape(dim_W * dim_U, dim_V)
                # Wait, T_vec index is k * dim_V * dim_U + j * dim_U + i
                # So T_vec.reshape(dim_W, dim_V, dim_U)
                # We want Y_V[k, i, j] = T[k, j, i]
                T_3d = T_vec.reshape(dim_W, dim_V, dim_U)
                Y_V = T_3d.transpose(0, 2, 1).reshape(dim_W * dim_U, dim_V)
                
                try:
                    V_new = np.linalg.lstsq(Z_V, Y_V, rcond=None)[0].T
                except:
                    break
                
                # Update W
                # T_{k, j, i} = sum_c W_{k, c} (U_{i, c} V_{j, c})
                # Y_W is (dim_U * dim_V, dim_W)
                # Z_W is (dim_U * dim_V, rank)
                # Z_W[i, j, c] = U[i, c] * V[j, c]
                Z_W_3d = U[:, np.newaxis, :] * V[np.newaxis, :, :]
                Z_W = Z_W_3d.reshape(dim_U * dim_V, rank)
                Y_W = T_3d.transpose(2, 1, 0).reshape(dim_U * dim_V, dim_W)
                
                try:
                    W_new = np.linalg.lstsq(Z_W, Y_W, rcond=None)[0].T
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check residual every 10 iterations
                if it % 10 == 0:
                    # Compute residual
                    # T_approx = sum_c U_c outer V_c outer W_c
                    # Reshape to (dim_W, dim_V, dim_U)
                    # U (dim_U, rank), V (dim_V, rank), W (dim_W, rank)
                    # T_approx[k, j, i] = sum_c W[k, c] V[j, c] U[i, c]
                    T_approx_3d = np.einsum('kc,jc,ic->kji', W, V, U)
                    T_approx_vec = T_approx_3d.reshape(-1)
                    residual = np.linalg.norm(T_vec - T_approx_vec)
                    
                    if residual < 1e-9:
                        found = True
                        break
            
            if found:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded version is still valid
                T_approx_3d = np.einsum('kc,jc,ic->kji', W_rounded, V_rounded, U_rounded)
                T_approx_vec = T_approx_3d.reshape(-1)
                residual_rounded = np.linalg.norm(T_vec - T_approx_vec)
                
                if residual_rounded < 1e-9:
                    # Success
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = rank
                    best_residual = 0
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                else:
                    # Store unrounded as backup if no better found?
                    # But task requires integer/half-integer.
                    # If we can't round, this rank might not be achievable with half-integers?
                    # Or we need better optimization.
                    # For now, keep searching.
                    pass
            else:
                # Check if residual was small but not 0
                if residual < 1e-5:
                    # Try rounding again
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    T_approx_3d = np.einsum('kc,jc,ic->kji', W_rounded, V_rounded, U_rounded)
                    T_approx_vec = T_approx_3d.reshape(-1)
                    residual_rounded = np.linalg.norm(T_vec - T_approx_vec)
                    if residual_rounded < 1e-9:
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = rank
                        best_residual = 0
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }

    # If no rank < 45 found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
