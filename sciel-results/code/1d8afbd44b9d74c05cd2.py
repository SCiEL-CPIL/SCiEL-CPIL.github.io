import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Target: Find rank-minimal decomposition for 3x3x8.
    # Strategy: Decompose into two 3x3x4 blocks.
    # Each 3x3x4 block = one 3x3x3 block + one 3x3x1 block.
    # Rank(3x3x1) is 3 (schoolbook).
    # Rank(3x3x3) is known to be 23.
    # So target rank = 2 * (23 + 3) = 52.
    
    # Step 1: Find a valid half-integer decomposition for 3x3x3 with rank <= 27.
    # We will try ranks 23, 24, 25, 26, 27.
    
    def get_tensor_333():
        # T[i, j, k] where i in 0..2, j in 0..2, k in 0..2
        # T[i, j, k] = 1 if indices match for C_ik = sum_j A_ij B_jk
        # Tensor index: (i*m+j, j*p+k, k*n+i)
        # For 3x3x3, n=m=p=3.
        # U_idx = i*3+j, V_idx = j*3+k, W_idx = k*3+i
        # T[U_idx, V_idx, W_idx] = 1 if the indices are consistent?
        # No, the tensor is defined on the product space.
        # The entry is 1 if the triple (i,j,k) is valid.
        # In the flattened form:
        T = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = i * 3 + j
                    v_idx = j * 3 + k
                    w_idx = k * 3 + i
                    T[u_idx, v_idx, w_idx] = 1
        return T

    T333 = get_tensor_333()
    
    def als_decompose(T, rank, max_iter=500, tol=1e-8):
        # Alternating Least Squares
        n_rows = T.shape[0]
        n_cols = T.shape[1]
        n_depth = T.shape[2]
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n_rows, rank)
            V = np.random.randn(n_cols, rank)
            W = np.random.randn(n_depth, rank)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U
                # T is (n_rows, n_cols, n_depth)
                # We want U such that U @ (V^T W^T) approx T
                # Flatten T to (n_rows, n_cols * n_depth)
                T_mat = T.reshape(n_rows, -1)
                # Compute V^T W^T? No, the product is sum_r V[:,r] W[:,r]^T
                # This is V @ W.T
                VW = V @ W.T
                # T_mat = U @ VW
                # U = T_mat @ VW^+
                try:
                    U_new = T_mat @ np.linalg.pinv(VW)
                except:
                    continue
                U = U_new
                
                # Update V
                # T is (n_cols, n_rows, n_depth) ?
                # We need T to be (n_cols, n_rows * n_depth)
                T_mat_v = np.transpose(T, (1, 0, 2)).reshape(n_cols, -1)
                UW = U @ W.T
                try:
                    V_new = T_mat_v @ np.linalg.pinv(UW)
                except:
                    continue
                V = V_new
                
                # Update W
                # T is (n_depth, n_rows, n_cols)
                T_mat_w = np.transpose(T, (2, 0, 1)).reshape(n_depth, -1)
                UV = U @ V.T
                try:
                    W_new = T_mat_w @ np.linalg.pinv(UV)
                except:
                    continue
                W = W_new
                
            # Calculate error
            # Reconstruct T_rec
            # T_rec[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # This is (U @ V.T) elementwise? No.
            # T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < tol:
                break
                
        return best_U, best_V, best_W, best_err

    def round_to_half_int(M):
        # Round to nearest 0.5
        return np.round(M * 2) / 2

    def check_exact(U, V, W, T, tol=1e-6):
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    U333, V333, W333, err333 = None, None, None, float('inf')
    rank333 = -1
    
    for r in range(23, 28):
        if time.time() > deadline:
            break
        U, V, W, err = als_decompose(T333, r)
        if U is not None:
            # Try rounding
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)
            
            if check_exact(U_r, V_r, W_r, T333):
                U333, V333, W333 = U_r, V_r, W_r
                rank333 = r
                break
                
    if rank333 == -1:
        # Fallback to schoolbook for 3x3x3 (rank 27)
        # Just use rank 27 schoolbook
        U333 = np.zeros((9, 27))
        V333 = np.zeros((9, 27))
        W333 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U333[i*3+j, col] = 1
                    V333[j*3+k, col] = 1
                    W333[k*3+i, col] = 1
                    col += 1
        rank333 = 27

    # Construct 3x3x8 decomposition
    # We have two blocks of 3x3x4.
    # Each 3x3x4 uses rank333 + 3.
    # Total rank = 2 * (rank333 + 3)
    
    total_rank = 2 * (rank333 + 3)
    
    U = np.zeros((9, total_rank))
    V = np.zeros((9, total_rank)) # 3x8 -> 3*8=24? No, V is (m*p, rank) = (3*8, rank) = (24, rank)
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank)) # W is (p*n, rank) = (8*3, rank) = (24, rank)
    
    # Block 1: columns 0,1,2,3 of B
    # Sub-block 1a: 3x3x3 (cols 0,1,2)
    # Sub-block 1b: 3x3x1 (col 3)
    
    # We need to map the indices of the 3x3x3 solution to the 3x3x8 context.
    # In 3x3x3:
    # U_idx = i*3+j
    # V_idx = j*3+k (k in 0..2)
    # W_idx = k*3+i
    
    # In 3x3x8:
    # U_idx = i*3+j
    # V_idx = j*8+k (k in 0..7)
    # W_idx = k*3+i
    
    # For sub-block 1a (k in 0..2):
    # U is same.
    # V: j*3+k in 3x3x3 -> j*8+k in 3x3x8.
    # W: k*3+i in 3x3x3 -> k*3+i in 3x3x8 (since k<3, k*3+i is same offset in first 9 rows of W? No, W is 24 rows. k*3+i is 0..8. Same.)
    
    # For sub-block 1b (k=3):
    # This is 3x3x1.
    # U: i*3+j.
    # V: j*8+3.
    # W: 3*3+i = 9+i.
    # Rank 3 schoolbook for this part.
    
    # Let's build the matrices.
    
    # 1a: 3x3x3 part
    # Columns 0 to rank333-1
    U[:, 0:rank333] = U333
    # V mapping: j*3+k -> j*8+k
    for j in range(3):
        for k in range(3):
            v_idx_333 = j * 3 + k
            v_idx_338 = j * 8 + k
            V[v_idx_338, 0:rank333] = V333[v_idx_333, :]
    # W mapping: k*3+i -> k*3+i (same)
    W[0:9, 0:rank333] = W333
    
    # 1b: 3x3x1 part (col 3)
    # Rank 3
    start_col = rank333
    for c in range(3):
        # c corresponds to i? No, schoolbook for 3x3x1:
        # C_i3 = sum_j A_ij B_j3
        # Terms: A_00 B_03, A_01 B_13, A_02 B_23 ...
        # U[i*3+j, c] = 1
        # V[j*8+3, c] = 1
        # W[3*3+i, c] = 1
        i = c
        for j in range(3):
            u_idx = i * 3 + j
            v_idx = j * 8 + 3
            w_idx = 9 + i
            U[u_idx, start_col + c] = 1
            V[v_idx, start_col + c] = 1
            W[w_idx, start_col + c] = 1
            
    # Block 2: columns 4,5,6,7 of B
    start_col_2 = 2 * rank333 + 3
    
    # 2a: 3x3x3 part (cols 4,5,6)
    # U is same.
    U[:, start_col_2:start_col_2+rank333] = U333
    # V mapping: j*3+k (k in 0..2) -> j*8+(k+4)
    for j in range(3):
        for k in range(3):
            v_idx_333 = j * 3 + k
            v_idx_338 = j * 8 + (k + 4)
            V[v_idx_338, start_col_2:start_col_2+rank333] = V333[v_idx_333, :]
    # W mapping: k*3+i (k in 0..2) -> (k+4)*3+i
    for k in range(3):
        for i in range(3):
            w_idx_333 = k * 3 + i
            w_idx_338 = (k + 4) * 3 + i
            W[w_idx_338, start_col_2:start_col_2+rank333] = W333[w_idx_333, :]
            
    # 2b: 3x3x1 part (col 7)
    start_col_2b = start_col_2 + rank333
    for c in range(3):
        i = c
        for j in range(3):
            u_idx = i * 3 + j
            v_idx = j * 8 + 7
            w_idx = 7 * 3 + i
            U[u_idx, start_col_2b + c] = 1
            V[v_idx, start_col_2b + c] = 1
            W[w_idx, start_col_2b + c] = 1

    # Convert to list of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
