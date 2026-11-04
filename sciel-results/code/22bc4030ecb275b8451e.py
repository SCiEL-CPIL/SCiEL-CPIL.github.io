import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    # 1. Construct the base rank-28 Strassen decomposition
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Define the 4 blocks of the 2x4 * 4x4 multiplication
    # Block 1: A_L * B_TL -> C_L (cols 0,1)
    # A_L indices: 0,1,4,5 (rows 0,1; cols 0,1)
    # B_TL indices: 0,1,4,5 (rows 0,1; cols 0,1)
    # C_L indices in W: 
    #   W[k*n+i]. C is 2x4. C_L is C[:, 0:2].
    #   C[0,0] -> W[0*2+0]=0
    #   C[0,1] -> W[1*2+0]=2
    #   C[1,0] -> W[0*2+1]=1
    #   C[1,1] -> W[1*2+1]=3
    #   So Z = [0, 2, 1, 3] maps local [c00, c01, c10, c11] to global W indices.
    #   Wait, local indices in add_block are X[0..3], Y[0..3], Z[0..3].
    #   X[0]=A00, X[1]=A01, X[2]=A10, X[3]=A11.
    #   Z[0]=C00, Z[1]=C01, Z[2]=C10, Z[3]=C11.
    #   Global W for C_L:
    #   C00 (global 0)
    #   C01 (global 2)
    #   C10 (global 1)
    #   C11 (global 3)
    #   So Z = [0, 2, 1, 3].

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A_R * B_BL -> C_L (cols 0,1)
    # A_R indices: 2,3,6,7 (rows 0,1; cols 2,3)
    # B_BL indices: 8,9,12,13 (rows 2,3; cols 0,1)
    # C_L indices in W: same as above [0, 2, 1, 3]
    
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A_L * B_TR -> C_R (cols 2,3)
    # A_L indices: 0,1,4,5
    # B_TR indices: 2,3,6,7 (rows 0,1; cols 2,3)
    # C_R indices in W:
    #   C[0,2] -> W[2*2+0]=4
    #   C[0,3] -> W[3*2+0]=6
    #   C[1,2] -> W[2*2+1]=5
    #   C[1,3] -> W[3*2+1]=7
    #   So Z = [4, 6, 5, 7].

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A_R * B_BR -> C_R (cols 2,3)
    # A_R indices: 2,3,6,7
    # B_BR indices: 10,11,14,15 (rows 2,3; cols 2,3)
    # C_R indices in W: same as above [4, 6, 5, 7]

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    # 2. Search for rank reduction via linear dependencies
    # We look for a vector c in R^28 such that:
    # sum c_k U[:,k] = 0
    # sum c_k V[:,k] = 0
    # sum c_k W[:,k] = 0
    # If such a c exists with non-zero entries, we can reduce rank.
    # This is equivalent to finding a null vector of the matrix M where M is formed by stacking U^T, V^T, W^T?
    # Actually, the condition is that the tensor T = sum U_k \otimes V_k \otimes W_k is zero?
    # No, we want to find a combination of the rank-1 terms that sums to zero.
    # T = \sum_{k=1}^{R} u_k v_k^T w_k^T.
    # We want \sum c_k u_k v_k^T w_k^T = 0.
    # This is a system of linear equations in c.
    # Let's construct the matrix A where each row corresponds to a triplet (i,j,k) and column to rank k.
    # A[i,j,k, r] = U[i,r] * V[j,r] * W[k,r].
    # We want A c = 0.
    
    # Dimensions: 8 * 16 * 8 = 1024 equations.
    # 28 unknowns.
    # We can use numpy.linalg.lstsq or svd.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to tensors
    # U: 8x28, V: 16x28, W: 8x28
    # We need to form the matrix A of size (8*16*8) x 28
    # A[ (i*16+j)*8+k, r ] = U[i,r] * V[j,r] * W[k,r]
    
    # Efficient construction using broadcasting
    # U_exp: 8 x 1 x 1 x 28
    # V_exp: 1 x 16 x 1 x 28
    # W_exp: 1 x 1 x 8 x 28
    # Product: 8 x 16 x 8 x 28
    
    U_exp = U_np[:, np.newaxis, np.newaxis, :]
    V_exp = V_np[np.newaxis, :, np.newaxis, :]
    W_exp = W_np[np.newaxis, np.newaxis, :, :]
    
    Tensor_Prod = U_exp * V_exp * W_exp
    A_matrix = Tensor_Prod.reshape(-1, 28)
    
    # Find null space of A_matrix
    # Use SVD
    try:
        U_svd, s, Vt = np.linalg.svd(A_matrix, full_matrices=False)
        # The null space is spanned by the last few rows of Vt corresponding to small singular values
        # Threshold for zero
        tol = 1e-9
        rank_est = np.sum(s > tol)
        
        if rank_est < 28:
            # There is a null space
            # The null vectors are the rows of Vt from index rank_est to 27
            null_vecs = Vt[rank_est:]
            
            # We want to eliminate as many terms as possible.
            # Ideally, we find a null vector c with many non-zero components.
            # Let's try to find a c that has large support.
            # We can iterate through null vectors and pick the one with largest support (number of non-zeros).
            # Or we can combine them.
            
            # For now, let's just check if we can reduce by 1.
            # If we have a null vector c, we can remove one term?
            # No, we can replace the basis.
            # If we have a linear dependency, we can express one term as a linear combination of others?
            # Actually, if \sum c_k T_k = 0, then T_{k0} = -1/c_{k0} \sum_{k \neq k0} c_k T_k.
            # So we can remove T_{k0} and adjust the coefficients of the others.
            # This reduces the rank by 1.
            
            # Let's find the best null vector to use.
            # We want a null vector c such that c has a component we can "remove".
            # Usually, we pick the component with largest magnitude to be the one removed.
            
            best_c = None
            best_support = 0
            for c in null_vecs:
                support = np.sum(np.abs(c) > tol)
                if support > best_support:
                    best_support = support
                    best_c = c
            
            if best_c is not None and best_support > 1:
                # Find index of max magnitude
                max_idx = np.argmax(np.abs(best_c))
                max_val = best_c[max_idx]
                
                if abs(max_val) > tol:
                    # Remove term max_idx
                    # New U, V, W will have rank 27
                    new_rank = 27
                    new_U = np.zeros((8, new_rank))
                    new_V = np.zeros((16, new_rank))
                    new_W = np.zeros((8, new_rank))
                    
                    # Copy all terms except max_idx
                    keep_indices = [i for i in range(28) if i != max_idx]
                    
                    for new_col, old_col in enumerate(keep_indices):
                        # The new coefficient for term old_col is best_c[old_col] / best_c[max_idx] ?
                        # Wait.
                        # \sum_{k} c_k T_k = 0
                        # c_{max} T_{max} + \sum_{k \neq max} c_k T_k = 0
                        # T_{max} = - \sum_{k \neq max} (c_k / c_{max}) T_k
                        # So we can replace T_{max} with the sum.
                        # The new decomposition is just the sum of the other terms, but scaled?
                        # No, the tensor is \sum_{k \neq max} c_k T_k - c_{max} T_{max}.
                        # If we just drop T_{max}, we lose c_{max} T_{max}.
                        # But we know c_{max} T_{max} = \sum_{k \neq max} (-c_k) T_k.
                        # So the total tensor is \sum_{k \neq max} c_k T_k + \sum_{k \neq max} (-c_k) T_k = 0?
                        # No.
                        # Original Tensor T = \sum_{k=1}^{28} T_k.
                        # We have a dependency \sum c_k T_k = 0.
                        # This means the T_k are linearly dependent.
                        # We can choose a new basis for the span of {T_k}.
                        # If we remove T_{max}, can we represent T_{max} as a combination of the others?
                        # Yes, T_{max} = -1/c_{max} \sum_{k \neq max} c_k T_k.
                        # So the span of {T_k | k \neq max} is the same as span of {T_k | all k}.
                        # However, the decomposition requires T = \sum_{new} U'_k V'_k W'_k.
                        # If we just drop T_{max}, we get T' = T - T_{max}. This is not T.
                        # We need to absorb T_{max} into the other terms.
                        # T = \sum_{k \neq max} T_k + T_{max}
                        # T = \sum_{k \neq max} T_k - \frac{1}{c_{max}} \sum_{k \neq max} c_k T_k
                        # T = \sum_{k \neq max} (1 - \frac{c_k}{c_{max}}) T_k
                        # So we can define new terms:
                        # T'_k = (1 - c_k/c_{max}) T_k for k \neq max.
                        # This scales the U, V, W vectors for term k by (1 - c_k/c_{max}).
                        
                        for i in range(8):
                            new_U[i, new_col] = U_np[i, old_col] * (1 - best_c[old_col] / max_val)
                        for j in range(16):
                            new_V[j, new_col] = V_np[j, old_col] * (1 - best_c[old_col] / max_val)
                        for k in range(8):
                            new_W[k, new_col] = W_np[k, old_col] * (1 - best_c[old_col] / max_val)
                    
                    # Verify the new decomposition
                    # Check if new_U, new_V, new_W produce the same tensor.
                    # Due to floating point errors, this might not be exact.
                    # But the problem allows integer/half-integer entries?
                    # "integer/half-integer entries"
                    # If the scaling factor is not integer/half-integer, we might violate the constraint.
                    # However, the Strassen coefficients are integers.
                    # If best_c has rational entries, the scaling might be rational.
                    # Let's check if the new entries are close to integers or half-integers.
                    
                    # For now, let's just return the rank-28 solution if the reduction is risky.
                    # The prompt says "score is the rank".
                    # If we can reduce to 27, score is 27.
                    # Let's verify the new tensor.
                    
                    U_check = np.array(new_U, dtype=float)
                    V_check = np.array(new_V, dtype=float)
                    W_check = np.array(new_W, dtype=float)
                    
                    # Reconstruct tensor
                    # This is expensive but we only do it once.
                    # T_new = sum_{k=1}^{27} U'_k V'_k W'_k
                    # T_old = sum_{k=1}^{28} U_k V_k W_k
                    # We know T_old is the matrix multiplication tensor.
                    # Let's check if T_new == T_old.
                    
                    # Efficient check:
                    # Compute T_new and T_old as vectors of length 1024.
                    # T_old_vec = A_matrix @ np.ones(28) ? No.
                    # A_matrix columns are T_k flattened.
                    # So T_old_vec = A_matrix @ np.ones(28).
                    # T_new_vec = A_matrix_new @ np.ones(27).
                    # A_matrix_new is A_matrix with column max_idx removed and scaled.
                    # Col k of A_matrix_new = (1 - c_k/c_max) * Col k of A_matrix.
                    # So T_new_vec = sum_{k \neq max} (1 - c_k/c_max) A_matrix[:, k].
                    # T_old_vec = sum_{k \neq max} A_matrix[:, k] + A_matrix[:, max].
                    # We know sum_{k} c_k A_matrix[:, k] = 0.
                    # So A_matrix[:, max] = -1/c_max sum_{k \neq max} c_k A_matrix[:, k].
                    # T_old_vec = sum_{k \neq max} A_matrix[:, k] - 1/c_max sum_{k \neq max} c_k A_matrix[:, k]
                    # T_old_vec = sum_{k \neq max} (1 - c_k/c_max) A_matrix[:, k]
                    # So T_new_vec should be exactly T_old_vec.
                    
                    # Let's verify numerically.
                    T_old_vec = A_matrix @ np.ones(28)
                    
                    # Construct T_new_vec
                    T_new_vec = np.zeros(1024)
                    for new_col, old_col in enumerate(keep_indices):
                        scale = 1 - best_c[old_col] / max_val
                        T_new_vec += scale * A_matrix[:, old_col]
                    
                    if np.allclose(T_old_vec, T_new_vec, atol=1e-9):
                        # Check if entries are valid (integer or half-integer)
                        # Multiply by 2 to check for integers
                        new_U_2 = new_U * 2
                        new_V_2 = new_V * 2
                        new_W_2 = new_W * 2
                        
                        # Check if close to integers
                        if (np.allclose(new_U_2, np.round(new_U_2), atol=1e-6) and
                            np.allclose(new_V_2, np.round(new_V_2), atol=1e-6) and
                            np.allclose(new_W_2, np.round(new_W_2), atol=1e-6)):
                            
                            # Convert to list of lists of floats (which are integers or .5)
                            # The problem asks for integer/half-integer entries.
                            # Returning floats that are exactly 1.0, 0.5, etc. should be fine.
                            
                            return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
                    else:
                        # Numerical error in verification, fallback
                        pass
                        
    except Exception:
        pass

    # Fallback to rank 28
    return {"U": U, "V": V, "W": W}
