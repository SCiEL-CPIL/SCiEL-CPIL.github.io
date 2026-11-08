import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    # Start with 56, try to reduce
    target_ranks = [52, 48, 44]
    
    # Get the baseline rank 56 solution
    U56, V56, W56 = _get_baseline_56()
    
    # Tensor T as a flat vector
    # T[i, j, k] = 1 if i==j and j==k? No.
    # T is the matrix multiplication tensor.
    # T[i, j, k] = delta_{i, j'} delta_{j', k'}? 
    # The tensor T satisfies C[i, k] = sum_j A[i, j] B[j, k].
    # So T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # T[i, j, k] = 1 if the indices match the multiplication structure.
    # Specifically, T is a 3-tensor of size (n*m, m*p, p*n)?
    # No, the interface is U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The decomposition is T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    # T[i, j, k] where i in 0..n*m-1, j in 0..m*p-1, k in 0..p*n-1.
    # i = i_A * m + j_A
    # j = j_B * p + k_B
    # k = k_C * n + i_C
    # T[i, j, k] = 1 if j_A == j_B and i_A == i_C.
    
    def get_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i_A in range(n):
            for j_A in range(m):
                for j_B in range(m):
                    for k_B in range(p):
                        if j_A == j_B:
                            i_C = i_A
                            k_C = k_B
                            i = i_A * m + j_A
                            j = j_B * p + k_B
                            k = k_C * n + i_C
                            T[i, j, k] = 1.0
        return T.flatten()

    T_flat = get_tensor(n, m, p)
    
    def compute_residual(U, V, W, R):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Residual = T_flat - sum_r U[:, r] outer V[:, r] outer W[:, r]
        # Efficiently:
        # T_approx = einsum('ir,jr,kr->ijk', U, V, W)
        # return T_flat - T_approx.flatten()
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return T_flat - T_approx.flatten()

    def objective(params, R):
        # params is flat vector of U, V, W
        n_m = n * m
        m_p = m * p
        p_n = p * n
        size_U = n_m * R
        size_V = m_p * R
        size_W = p_n * R
        
        U = params[0:size_U].reshape(n_m, R)
        V = params[size_U:size_U+size_V].reshape(m_p, R)
        W = params[size_U+size_V:size_U+size_V+size_W].reshape(p_n, R)
        
        res = compute_residual(U, V, W, R)
        return np.sum(res**2)

    best_result = None
    best_score = float('inf')
    
    # Try to reduce rank
    for R in target_ranks:
        # Initialize with subset of rank 56 factors
        # Randomly pick R factors from the 56
        idx = np.random.choice(56, R, replace=False)
        U_init = U56[:, idx]
        V_init = V56[:, idx]
        W_init = W56[:, idx]
        
        # Flatten
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize
        # Use Nelder-Mead or Powell for small R?
        # R=48, variables ~ 2600. Too many for Nelder-Mead.
        # Use BFGS or L-BFGS-B.
        try:
            result = minimize(objective, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if result.success or result.fun < 1e-6:
                # Extract U, V, W
                n_m = n * m
                m_p = m * p
                p_n = p * n
                U_opt = result.x[0:n_m*R].reshape(n_m, R)
                V_opt = result.x[n_m*R:n_m*R+m_p*R].reshape(m_p, R)
                W_opt = result.x[n_m*R+m_p*R:n_m*R+m_p*R+p_n*R].reshape(p_n, R)
                
                # Check if close to half-integers
                # Multiply by 2
                U2 = U_opt * 2
                V2 = V_opt * 2
                W2 = W_opt * 2
                
                # Check distance to nearest integer
                dist_U = np.max(np.abs(U2 - np.round(U2)))
                dist_V = np.max(np.abs(V2 - np.round(V2)))
                dist_W = np.max(np.abs(W2 - np.round(W2)))
                
                if dist_U < 1e-5 and dist_V < 1e-5 and dist_W < 1e-5:
                    # Snap to half-integers
                    U_final = np.round(U2) / 2.0
                    V_final = np.round(V2) / 2.0
                    W_final = np.round(W2) / 2.0
                    
                    # Verify
                    res = compute_residual(U_final, V_final, W_final, R)
                    if np.max(np.abs(res)) < 1e-9:
                        # Convert to lists
                        U_list = U_final.tolist()
                        V_list = V_final.tolist()
                        W_list = W_final.tolist()
                        
                        # Check if better than current best
                        if R < best_score:
                            best_score = R
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass

    if best_result:
        return best_result
    else:
        # Fallback to rank 56
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}

def _get_baseline_56():
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    # 8 blocks of 2x2x2
    for i_off in [0]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4, 6]:
                # Calculate base rank
                # Order: i_off (1), j_off (2), k_off (4)
                # base = i_off*32 + j_off*16 + k_off*4 ?
                # Let's just use a counter
                pass

    # Simpler: loop through all 8 blocks
    block_idx = 0
    for i_off in [0]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4, 6]:
                add_block(block_idx * 7, i_off, j_off, k_off)
                block_idx += 1

    return U, V, W
# EVOLVE-BLOCK-END
