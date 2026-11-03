import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = 1
        U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[0, 3] = -1; W[1, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = 1; W[2, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[2, 5] = 1; W[2, 5] = 1; W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1; V[1, 6] = 1; V[3, 6] = 1; W[0, 6] = 1; W[1, 6] = 1
        return U, V, W

    def get_mult_1x2_2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Baseline Rank 66 Construction
    U_66 = np.zeros((12, 66))
    V_66 = np.zeros((24, 66))
    W_66 = np.zeros((18, 66))
    
    def place_block(U_block, V_block, W_block, u_rows, v_rows, w_rows, offset):
        rank_block = U_block.shape[1]
        for c in range(rank_block):
            for local_r, global_r in enumerate(u_rows):
                U_66[global_r, offset + c] = U_block[local_r, c]
            for local_r, global_r in enumerate(v_rows):
                V_66[global_r, offset + c] = V_block[local_r, c]
            for local_r, global_r in enumerate(w_rows):
                W_66[global_r, offset + c] = W_block[local_r, c]

    S_U, S_V, S_W = get_strassen_2x2()
    M12_U, M12_V, M12_W = get_mult_1x2_2x2()
    
    def get_u_idx(i_start, i_end, j_start, j_end):
        return [i*m + j for i in range(i_start, i_end) for j in range(j_start, j_end)]
    def get_v_idx(j_start, j_end, k_start, k_end):
        return [j*p + k for j in range(j_start, j_end) for k in range(k_start, k_end)]
    def get_w_idx(k_start, k_end, i_start, i_end):
        return [k*n + i for k in range(k_start, k_end) for i in range(i_start, i_end)]

    col_offset = 0
    # 6 blocks of 2x2 Strassen + 1x2 blocks
    for j_block in range(2):
        for k_block in range(3):
            j_start = j_block * 2
            j_end = j_block * 2 + 2
            k_start = k_block * 2
            k_end = k_block * 2 + 2
            u_rows = get_u_idx(0, 2, j_start, j_end)
            v_rows = get_v_idx(j_start, j_end, k_start, k_end)
            w_rows = get_w_idx(k_start, k_end, 0, 2)
            place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
            
            u_rows = get_u_idx(2, 3, j_start, j_end)
            v_rows = get_v_idx(j_start, j_end, k_start, k_end)
            w_rows = get_w_idx(k_start, k_end, 2, 3)
            place_block(M12_U, M12_V, M12_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 4

    total_rank = col_offset
    U_curr = U_66[:, :total_rank].copy()
    V_curr = V_66[:, :total_rank].copy()
    W_curr = W_66[:, :total_rank].copy()

    if verify(U_curr.tolist(), V_curr.tolist(), W_curr.tolist(), n, m, p):
        best_U = U_curr.tolist()
        best_V = V_curr.tolist()
        best_W = W_curr.tolist()
        best_rank = total_rank

    # Structural Change: Rank-Reduction Search via Tensor Nullspace
    # We represent the problem as finding coefficients x (size 65) such that:
    # sum_{c=0}^{64} x_c (U_c \otimes V_c \otimes W_c) = T
    # This is a linear system A x = b, where A is (72x65) and b is the flattened multiplication tensor.
    # If a solution exists, we can form a rank-65 decomposition by forming linear combinations of the basis tensors.
    # Note: The new columns will be linear combinations of the old ones.
    
    def try_reduce_rank(U, V, W, n, m, p, target_rank):
        if time.time() > deadline - 10:
            return None
            
        cur_rank = U.shape[1]
        if target_rank >= cur_rank:
            return U, V, W
            
        # Flatten tensors into vectors of size n*m*p
        # We keep the first target_rank columns as the "basis" for the new decomposition?
        # No, we want to see if T is in the span of ANY subset of size target_rank from the cur_rank available.
        # That is expensive.
        
        # Instead, let's try to "re-express" T using the first target_rank columns.
        # Construct design matrix M where column c is the flattening of the tensor T_c.
        # We want M[:, :target_rank] x = T_flat.
        
        dim_t = n * m * p
        M = np.zeros((dim_t, cur_rank))
        
        for c in range(cur_rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            
            # T_c[i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We need to fill M flat index corresponding to (i, j, k) with this value.
            # Flat index for (i, j, k) in order i, j, k is i * (m*p) + j * p + k.
            
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if u_val == 0: continue
                    base_u = i * m + j
                    for k in range(p):
                        v_val = v_col[j * p + k]
                        if v_val == 0: continue
                        w_val = w_col[k * n + i]
                        if w_val == 0: continue
                        val = u_val * v_val * w_val
                        flat_idx = i * (m * p) + j * p + k
                        M[flat_idx, c] = val
            
        T_flat = np.ones(dim_t)
        
        # Solve least squares for the first target_rank columns
        # M_sub = M[:, :target_rank]
        # x, residuals, rank, s = np.linalg.lstsq(M_sub, T_flat, rcond=None)
        
        # Check if the residual is small.
        # If yes, we have x such that sum_{c < target_rank} x_c T_c \approx T.
        # Then we can define new U', V', W' where column c is x_c * (U_col_c, V_col_c, W_col_c).
        
        try:
            M_sub = M[:, :target_rank]
            # If rank of M_sub is target_rank, we can check if T is in the column space.
            # Using lstsq is a good heuristic for exactness if residual is ~0.
            x, residuals, rank_solve, s = np.linalg.lstsq(M_sub, T_flat, rcond=None)
            
            if len(residuals) > 0 and residuals[0] < 1e-6:
                # Success! We found coefficients.
                # Check exactness
                reconstructed = M_sub @ x
                if np.allclose(reconstructed, T_flat, atol=1e-9):
                    U_new = np.zeros((U.shape[0], target_rank))
                    V_new = np.zeros((V.shape[0], target_rank))
                    W_new = np.zeros((W.shape[0], target_rank))
                    
                    for c in range(target_rank):
                        U_new[:, c] = U[:, c] * x[c]
                        V_new[:, c] = V[:, c] * x[c]
                        W_new[:, c] = W[:, c] * x[c]
                    
                    # Verify the new decomposition
                    if verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                        return U_new, V_new, W_new
                    else:
                        # Floating point issues? Try to snap to nearest integer/half-integer?
                        # The task allows integer/half-integer.
                        # If the solution x has nice properties, snapping might work.
                        # But for now, if verify fails due to precision, we might still accept it 
                        # if the residual was tiny, but verify is strict.
                        # Let's try to round x if it's close to integers?
                        # x might be something like 1.0000000001.
                        pass
        except Exception:
            pass
        
        return None

    # Try reducing from 66 to 65
    # This is a "structural change" because it attempts to find a fundamentally lower rank decomposition
    # using the linear algebra structure of the existing solution space.
    
    if time.time() < deadline - 20:
        result_65 = try_reduce_rank(U_curr, V_curr, W_curr, n, m, p, 65)
        if result_65:
            U_n, V_n, W_n = result_65
            best_U = U_n.tolist()
            best_V = V_n.tolist()
            best_W = W_n.tolist()
            best_rank = 65
            
            # Can we go lower?
            if time.time() < deadline - 5:
                 result_64 = try_reduce_rank(U_n, V_n, W_n, n, m, p, 64)
                 if result_64:
                    U_n, V_n, W_n = result_64
                    best_U = U_n.tolist()
                    best_V = V_n.tolist()
                    best_W = W_n.tolist()
                    best_rank = 64

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }

BET: {mechanism: "Structural switch from heuristic block-assembly to algebraic rank-reduction via nullspace search of the tensor decomposition matrix; this directly tests if the 66-rank basis is overcomplete for the target tensor, potentially dropping rank to 65.", q: 0.4, delta: [0.0, 0.1]}
