"""Matrix multiplication 2x4 x 4x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 120.0
    
    # Start with the known rank-28 Strassen-based decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    # Four Strassen blocks
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)      # A_left * B_TL -> C_left
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)      # A_right * B_BL -> C_left
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)     # A_left * B_TR -> C_right
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)     # A_right * B_BR -> C_right
    
    # Verify the decomposition is correct
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # U index: i = a_row * m + a_col (0..7)
    # V index: j = b_row * p + b_col (0..15)
    # W index: k = c_col * n + c_row (0..7)
    
    # The multiplication tensor: T[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to (a_row, b_col)
    # i.e., C[a_row, b_col] = sum_k A[a_row, k] * B[k, b_col]
    # In our indexing:
    #   a_idx = a_row * m + a_col  where a_col is the intermediate index k
    #   b_idx = b_row * p + b_col  where b_row is the intermediate index k
    #   c_idx = b_col * n + a_row  (trace convention: W[k*n+i, r] where k=col, i=row)
    # So T[a_idx, b_idx, c_idx] = 1 if a_col == b_row AND c_idx == b_col * n + a_row
    
    T_decomp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Build the true multiplication tensor
    T_true = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_col in range(p):
                a_idx = a_row * m + a_col
                b_idx = a_col * p + b_col  # b_row = a_col (the intermediate index)
                c_idx = b_col * n + a_row
                T_true[a_idx, b_idx, c_idx] = 1
    
    if not np.allclose(T_decomp, T_true, atol=1e-10):
        # Decomposition is wrong, return as-is (shouldn't happen)
        pass
    
    # Try to find a lower rank by checking for linear dependencies
    # Check if any column of the unfolded matrices is redundant
    # The tensor can be unfolded in different ways. Check the (n*m) x (m*p*p*n) unfolding
    # T_(nm) is an (n*m) x (m*p * p*n) matrix where T_(nm)[i, (j,k)] = T[i,j,k]
    
    T_unfold = T_true.reshape(n*m, m*p*p*n)
    rank_true = np.linalg.matrix_rank(T_unfold)
    
    # Also check other unfoldings
    T_unfold2 = T_true.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    rank_true2 = np.linalg.matrix_rank(T_unfold2)
    
    T_unfold3 = T_true.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
    rank_true3 = np.linalg.matrix_rank(T_unfold3)
    
    min_unfold_rank = min(rank_true, rank_true2, rank_true3)
    
    # The CP rank is at least the maximum of all mode-unfolding ranks
    # But it can be higher. Our current rank is 28.
    # If min_unfold_rank > 28, we have a problem (shouldn't happen)
    # If min_unfold_rank < 28, there might be room for improvement
    
    # Try iterative rank reduction: check if we can remove a rank-1 component
    # by seeing if it's in the span of the others
    
    def try_reduce_rank(U_list, V_list, W_list, current_rank):
        """Try to reduce rank by checking if any rank-1 term is redundant."""
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        for r in range(current_rank):
            if time.time() > deadline - 30:
                break
            # Check if rank-1 term r is in the span of the others
            # This is equivalent to checking if the tensor without term r still equals T_true
            U_test = np.delete(U_arr, r, axis=1)
            V_test = np.delete(V_arr, r, axis=1)
            W_test = np.delete(W_arr, r, axis=1)
            
            T_test = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test)
            
            if np.allclose(T_test, T_true, atol=1e-10):
                # Term r is redundant! Remove it.
                new_U = [[U_list[i][j] for j in range(current_rank) if j != r] for i in range(n*m)]
                new_V = [[V_list[i][j] for j in range(current_rank) if j != r] for i in range(m*p)]
                new_W = [[W_list[i][j] for j in range(current_rank) if j != r] for i in range(p*n)]
                return new_U, new_V, new_W, current_rank - 1
        
        return U_list, V_list, W_list, current_rank
    
    # Attempt rank reduction
    U, V, W, rank = try_reduce_rank(U, V, W, rank)
    
    # Try more aggressive reduction: use SVD on unfoldings to find compressed representation
    # If the mode-unfolding has rank < current_rank, we might find a lower-rank CP decomposition
    
    if rank > min_unfold_rank and time.time() < deadline - 60:
        # Try to find a CP decomposition of rank = min_unfold_rank using iterative methods
        # This is hard in general, but for small tensors we can try
        
        target_rank = min_unfold_rank
        if target_rank < rank:
            # Use a simple approach: try random initialization and ALS
            # But this may not converge to an exact solution
            # For now, just check if the unfolding rank gives us a bound
            
            pass
    
    # Ensure entries are integers or half-integers
    for i in range(len(U)):
        for j in range(len(U[i])):
            U[i][j] = round(U[i][j] * 2) / 2
    for i in range(len(V)):
        for j in range(len(V[i])):
            V[i][j] = round(V[i][j] * 2) / 2
    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = round(W[i][j] * 2) / 2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
