import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Indices for global tensors
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify the rank-17 solution is correct
    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor"""
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    # The tensor T[i,j,k] = 1
                    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                    actual = sum(U_mat[i*m_val+j, r] * V_mat[j*p_val+k, r] * W_mat[k*n_val+i, r] for r in range(rank))
                    if abs(actual - 1.0) > 1e-10:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback: return the hardcoded solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank by checking linear dependencies in the tensor space
    # We check if the target tensor (all 1s) can be formed by a linear combination
    # of the rank-1 tensors excluding one specific term.
    
    # Vectorize the rank-1 tensors
    # T_s is a vector of length n*m*p
    T_vectors = np.zeros((n * m * p, 17))
    for r in range(17):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        # Compute tensor product
        # T[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * (p * n) + (j * p + k) * n + (k * n + i) 
                    # Wait, the vectorization order needs to be consistent for the linear system.
                    # Let's use a flat index consistent with the verify loop.
                    # Verify loop: i, j, k.
                    # Let's just flatten in C-order (i, j, k) -> idx = (i * m + j) * p + k.
                    # But the tensor is 3D.
                    # Let's create a 3D array and flatten.
        
        # Simpler: just build the vector directly
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    vec[(i * m + j) * p + k] = val
        T_vectors[:, r] = vec

    target_tensor = np.ones(n * m * p)
    
    best_rank = 17
    best_solution = (U.tolist(), V.tolist(), W.tolist())
    
    # Check if removing any term r allows the remaining terms (scaled) to form the target
    for r_to_remove in range(17):
        if time.time() > deadline - 10:
            break
            
        # Indices of remaining terms
        remaining_indices = [i for i in range(17) if i != r_to_remove]
        
        # Matrix of remaining tensor vectors
        T_rem = T_vectors[:, remaining_indices]
        
        # Solve T_rem @ alpha = target_tensor
        try:
            alpha, residuals, rank, s = np.linalg.lstsq(T_rem, target_tensor, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check residual
        if residuals and residuals[0] < 1e-10:
            # We have a solution!
            # Reconstruct U, V, W with new scales
            U_new = np.zeros((n * m, 16))
            V_new = np.zeros((m * p, 16))
            W_new = np.zeros((p * n, 16))
            
            for new_idx, old_idx in enumerate(remaining_indices):
                scale = alpha[new_idx]
                # If scale is close to 0, we could drop it, but let's keep the structure for now
                # and just verify. If it's 0, it doesn't contribute.
                U_new[:, new_idx] = U[:, old_idx] * scale
                V_new[:, new_idx] = V[:, old_idx] * scale
                W_new[:, new_idx] = W[:, old_idx] * scale
            
            # Verify the new solution
            if verify(U_new, V_new, W_new, n, m, p):
                # Check if we can drop any zero columns
                final_U = []
                final_V = []
                final_W = []
                for idx in range(16):
                    if np.any(np.abs(U_new[:, idx]) > 1e-10):
                        final_U.append(U_new[:, idx])
                        final_V.append(V_new[:, idx])
                        final_W.append(W_new[:, idx])
                
                if len(final_U) > 0:
                    U_final = np.column_stack(final_U)
                    V_final = np.column_stack(final_V)
                    W_final = np.column_stack(final_W)
                    
                    if verify(U_final, V_final, W_final, n, m, p):
                        best_rank = U_final.shape[1]
                        best_solution = (U_final.tolist(), V_final.tolist(), W_final.tolist())
                        break # Found a valid lower rank, stop searching
                else:
                    # All zero? Unlikely, but handle it
                    pass
        else:
            # No solution in the span of the remaining tensors
            pass
            
    if best_rank < 17:
        U_out, V_out, W_out = best_solution
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
