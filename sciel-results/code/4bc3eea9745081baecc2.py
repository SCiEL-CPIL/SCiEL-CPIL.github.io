import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0

    # Rank 98 Strassen-based decomposition
    rank = 98

    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 decomposition details
    # U_2, V_2, W_2 are 4x7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # A_specs: How to form A_eff for each P_k
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]

    # B_specs: How to form B_eff for each P_k
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]

    # C_specs: How P_k contributes to C blocks
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]

    # Construct the rank-98 decomposition
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s

                # --- U (Input A) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]

                # --- V (Input B) ---
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]

                # --- W (Output C) ---
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]

    # Verify the decomposition is correct using vectorized approach
    def verify(U, V, W, n, m, p):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute via decomposition: C[i,k] = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k]) * W[k*n+i,r]
        # Vectorized: for each r, compute scalar_u[i] = sum_j U[i*m+j,r]*A[i,j], scalar_v[k] = sum_j V[j*p+k,r]*B[j,k]
        # Then C[i,k] += scalar_u[i] * scalar_v[k] * W[k*n+i,r]
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_col = U[:, r]  # (n*m,)
            v_col = V[:, r]  # (m*p,)
            w_col = W[:, r]  # (p*n,)
            
            # scalar_u[i] = sum_j u_col[i*m+j] * A[i,j]
            # Reshape u_col to (n, m)
            u_mat = u_col.reshape(n, m)
            scalar_u = np.sum(u_mat * A, axis=1)  # (n,)
            
            # scalar_v[k] = sum_j v_col[j*p+k] * B[j,k]
            # Reshape v_col to (m, p)
            v_mat = v_col.reshape(m, p)
            scalar_v = np.sum(v_mat * B, axis=0)  # (p,)
            
            # w_col indexed by k*n+i, reshape to (p, n)
            w_mat = w_col.reshape(p, n)
            # C[i,k] += scalar_u[i] * scalar_v[k] * w_mat[k, i]
            C_decomp += np.outer(scalar_u, scalar_v) * w_mat.T
        
        err = np.max(np.abs(C_decomp - C_expected))
        return err

    err = verify(U, V, W, n, m, p)
    if err > 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find linear dependencies among the rank-98 columns using SVD
    # Check if we can reduce rank by finding null space vectors
    
    # Build the "flattened" tensor matrix to check for dependencies
    # The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For each (i,j,k), we have: T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
    
    # Alternative: check if any column is a linear combination of others
    # Use SVD on U to find its null space, then check if those vectors also null out V and W
    
    # Get null space of U
    Ut = U.T  # (rank, n*m)
    _, S_U, Vt_U = np.linalg.svd(Ut, full_matrices=False)
    
    # The null space of U^T (which is the null space of U as a map from R^rank to R^{n*m})
    # Actually, we want vectors alpha in R^rank such that U @ alpha = 0
    # This is the null space of U (n*m x rank)
    # SVD of U: U = U_s @ S @ Vt, null space is span of rows of Vt where S ~ 0
    rank_U = np.sum(S_U > 1e-10)
    if rank_U < rank:
        null_U = Vt_U[rank_U:, :].T  # (rank, rank-rank_U)
        # Check each null vector against V and W
        for i in range(null_U.shape[1]):
            alpha = null_U[:, i]
            if np.max(np.abs(V @ alpha)) < 1e-8 and np.max(np.abs(W @ alpha)) < 1e-8:
                # Found a reducible direction, but we'd need to handle the tensor structure
                # For now, just note it
                pass
    
    # Similarly for V and W
    Vt = V.T
    _, S_V, Vt_V = np.linalg.svd(Vt, full_matrices=False)
    rank_V = np.sum(S_V > 1e-10)
    
    Wt = W.T
    _, S_W, Vt_W = np.linalg.svd(Wt, full_matrices=False)
    rank_W = np.sum(S_W > 1e-10)
    
    # The minimum possible rank is at least max(rank_U, rank_V, rank_W)
    # But this is a lower bound, not necessarily achievable
    
    # Try greedy column removal: remove one column at a time if the decomposition still holds
    def check_decomposition_fast(U, V, W, n, m, p, skip_cols=None):
        if skip_cols is None:
            skip_cols = set()
        
        # Use vectorized check
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            if r in skip_cols:
                continue
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            u_mat = u_col.reshape(n, m)
            scalar_u = np.sum(u_mat * A, axis=1)
            
            v_mat = v_col.reshape(m, p)
            scalar_v = np.sum(v_mat * B, axis=0)
            
            w_mat = w_col.reshape(p, n)
            C_decomp += np.outer(scalar_u, scalar_v) * w_mat.T
        
        err = np.max(np.abs(C_decomp - C_expected))
        return err
    
    # Try removing each column
    reducible_cols = []
    for r in range(rank):
        if time.time() > deadline - 20:
            break
        skip = {r}
        err = check_decomposition_fast(U, V, W, n, m, p, skip)
        if err < 1e-10:
            reducible_cols.append(r)
    
    if reducible_cols:
        new_cols = [r for r in range(rank) if r not in reducible_cols[0]]
        U_new = U[:, new_cols]
        V_new = V[:, new_cols]
        W_new = W[:, new_cols]
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
