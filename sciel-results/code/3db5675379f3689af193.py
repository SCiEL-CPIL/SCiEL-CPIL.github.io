DIMENSIONS = [2, 4, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition patterns
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
    
    # 8 blocks of 2x2x2 Strassen
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Verify baseline is correct
    def verify_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Check if decomposition is exact: sum_r U[i,r]*V[j,r]*W[k,r] = delta(i,j,k)
        where i indexes (i_row, i_col) in n*m, j indexes (j_row, j_col) in m*p,
        k indexes (k_row, k_col) in p*n
        The condition is: for all i_row, j_col: sum over middle index
        Actually the tensor condition is:
        T[i_row, j_col, k_row, k_col] = delta(i_row, k_row) * delta(j_col, k_col)
        which equals sum_r U[i_row*m+j_col, r] * V[j_row*p+k_col, r] * W[k_row*n+i_row, r]
        Wait, let me re-read the trace convention.
        
        Output W[k*n+i, r] means W is indexed by (k, i) where k is output row (0..p-1), i is output col (0..n-1)
        Actually for 2x4 * 4x8: output is 2x8
        So W should be indexed by (output_row, output_col) = (k, i) where k in 0..p-1=0..7, i in 0..n-1=0..1
        But W has shape (p*n, rank) = (16, rank)
        W[k*n+i, r] means index = k*2 + i for k in 0..7, i in 0..1
        
        U has shape (n*m, rank) = (8, rank), indexed by (i_row, i_col) = (i, j) where i in 0..1, j in 0..3
        So U[i*m+j, r] = U[i*4+j, r]
        
        V has shape (m*p, rank) = (32, rank), indexed by (j_row, j_col) = (j, k) where j in 0..3, k in 0..7
        So V[j*p+k, r] = V[j*8+k, r]
        
        The matrix product C[i,k] = sum_j A[i,j] * B[j,k]
        Tensor: T[i,j,k] = delta(i,k_row?)... 
        Actually the multiplication tensor T has T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
        So T[i,j,k] = 1 always (for valid i,j,k)
        
        The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        Where W is indexed by (k,i) with the trace convention W[k*n+i, r]
        
        So we need: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j, :]
                    v_val = V_arr[j * p + k, :]
                    w_val = W_arr[k * n + i, :]
                    s = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to reduce rank by checking if any rank term is linearly dependent
    def try_reduce_rank(U_mat, V_mat, W_mat, rank_val):
        """Try to eliminate linearly dependent terms.
        For each rank index r, check if removing it and adjusting others can still satisfy the equations.
        This is equivalent to checking if the r-th column is in the span of others in a certain sense.
        
        A simpler approach: for each pair of rank indices (r1, r2), check if we can eliminate one.
        Actually, a term r is removable if there exist adjustments to other terms such that the tensor is preserved.
        
        Let's use the approach: for each rank r, try to express its contribution as a combination of others.
        The contribution of term r is: outer(U[:,r], V[:,r]) ⊗ W[:,r] (in tensor form)
        
        More practically: for each r, check if the tensor T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] (as a 3D array)
        can be written as sum_{r' != r} c_{r'} * T_{r'}.
        
        This is a linear system. Let's set it up.
        """
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        
        # Build the 3D tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want to check if for some r, T without r can be adjusted to match T.
        
        # For each candidate rank r to remove:
        for r_remove in range(rank_val):
            if time.time() > deadline - 5:
                break
            
            # Build the "target" tensor contribution from term r_remove
            # T_target[i,j,k] = U[i*m+j, r_remove] * V[j*p+k, r_remove] * W[k*n+i, r_remove]
            # We need: sum_{r != r_remove} c_r * T_r[i,j,k] = T_target[i,j,k]
            # where T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is a linear system in the unknowns c_r (for r != r_remove)
            # Number of equations: n*m*p = 64
            # Number of unknowns: rank_val - 1 = 55 (for rank 56)
            # So the system is overdetermined (64 eqs, 55 unknowns) - likely no solution
            
            # Instead, let's try a different approach: try to find a linear combination of terms that equals zero
            # i.e., find coefficients c_r such that sum_r c_r * T_r = 0 and at least one c_r is nonzero.
            # If such exists, we can use it to eliminate one term.
            
            # Build the 64xR matrix where column r is the vectorized T_r
            R = rank_val
            A = np.zeros((n * m * p, R), dtype=np.float64)
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * (m * p) + j * p + k
                            A[idx, r] = U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
            
            # Find null space of A
            try:
                _, s, Vh = np.linalg.svd(A, full_matrices=True)
            except np.linalg.LinAlgError:
                continue
            
            # Find the smallest singular value
            if len(s) == 0:
                continue
            
            min_s = min(s)
            if min_s < 1e-10:
                # There's a null vector, meaning there's a linear dependence
                # The null vector gives coefficients c_r such that sum_r c_r * T_r = 0
                # Find the index with largest |c_r|
                null_vec = Vh[-1]  # Corresponds to smallest singular value
                max_idx = np.argmax(np.abs(null_vec))
                c_max = null_vec[max_idx]
                
                if abs(c_max) > 1e-10:
                    # We can eliminate term max_idx by adjusting others
                    # New U[:, max_idx] = 0, and for r != max_idx:
                    # U[:, r] += (c_r / c_max) * U[:, max_idx]
                    # Similarly for V and W
                    new_rank = R - 1
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    
                    new_r_idx = 0
                    for r in range(R):
                        if r == max_idx:
                            continue
                        # Adjust by the null vector
                        coeff = null_vec[r] / c_max
                        for i in range(n * m):
                            new_U[i][new_r_idx] = U_mat[i][r] + coeff * U_mat[i][max_idx]
                        for j in range(m * p):
                            new_V[j][new_r_idx] = V_mat[j][r] + coeff * V_mat[j][max_idx]
                        for k in range(p * n):
                            new_W[k][new_r_idx] = W_mat[k][r] + coeff * W_mat[k][max_idx]
                        new_r_idx += 1
                    
                    # Verify
                    if verify_decomposition(new_U, new_V, new_W, new_rank):
                        return new_U, new_V, new_W, new_rank
        return None
    
    # Attempt rank reduction
    best_result = (U, V, W, rank)
    current_U, current_V, current_W, current_rank = U, V, W, rank
    
    max_iterations = 10
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        result = try_reduce_rank(current_U, current_V, current_W, current_rank)
        if result is not None:
            new_U, new_V, new_W, new_rank = result
            current_U, current_V, current_W, current_rank = new_U, new_V, new_W, new_rank
            best_result = (current_U, current_V, current_W, current_rank)
            print(f"Reduced rank to {current_rank}")
        else:
            break
    
    U_final, V_final, W_final, rank_final = best_result
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
