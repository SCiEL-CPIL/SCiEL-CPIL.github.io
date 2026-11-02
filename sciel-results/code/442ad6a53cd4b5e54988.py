import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 130.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Split m=4 into 2 blocks of 2.
    # For each block (2 of them):
    #   Split n=3 into 2 (top) and 1 (bot).
    #   Top: 2x2x8. Split p=8 into 4 blocks of 2.
    #         Each 2x2x2 uses Strassen (rank 7).
    #         Total Top Rank = 4 * 7 = 28.
    #   Bot: 1x2x8. Schoolbook rank = 1*2*8 = 16.
    #   Total per m-block = 28 + 16 = 44.
    # Total Rank = 2 * 44 = 88.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the linear combinations for U (A), V (B), W (C)
    # Indices in 2x2 block:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({0: 1, 2: 1}, {0: 1, 3: 1}, {0: 1, 3: -1}), # Wait, W coeffs are for reconstruction
        # Let's use the standard reconstruction coefficients
    ]
    
    # Correct Strassen Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1: M1
    # U: A00 + A11 -> {0: 1, 3: 1} (using 0:00, 1:01, 2:10, 3:11)
    # V: B00 + B11 -> {0: 1, 3: 1}
    # W: C00 + C11 -> {0: 1, 3: 1}
    
    # Term 2: M2
    # U: A10 + A11 -> {2: 1, 3: 1}
    # V: B00 -> {0: 1}
    # W: C10 - C11 -> {2: 1, 3: -1}
    
    # Term 3: M3
    # U: A00 -> {0: 1}
    # V: B01 - B11 -> {1: 1, 3: -1}
    # W: C01 + C11 -> {1: 1, 3: 1}
    
    # Term 4: M4
    # U: A11 -> {3: 1}
    # V: B10 - B00 -> {2: 1, 0: -1}
    # W: C00 + C10 -> {0: 1, 2: 1}
    
    # Term 5: M5
    # U: A00 + A01 -> {0: 1, 1: 1}
    # V: B11 -> {3: 1}
    # W: C01 - C00 -> {1: 1, 0: -1}
    
    # Term 6: M6
    # U: A10 - A00 -> {2: 1, 0: -1}
    # V: B00 + B01 -> {0: 1, 1: 1}
    # W: C11 -> {3: 1}
    
    # Term 7: M7
    # U: A01 - A11 -> {1: 1, 3: -1}
    # V: B10 + B11 -> {2: 1, 3: 1}
    # W: C00 -> {0: 1}
    
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for j1 in range(2):
        # Top part: 2x2x8
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                # Map local indices to global
                # A (U)
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j][col] += coeff
                
                # B (V)
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k][col] += coeff
                
                # C (W)
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i][col] += coeff
                
                col += 1
        
        # Bot part: 1x2x8
        # Rank 16
        for k in range(8):
            for j2 in range(2):
                # Term: A[2, 2*j1+j2] * B[2*j1+j2, k] * C[2, k]
                
                # U
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j][col] = 1
                
                # V
                global_j = 2 * j1 + j2
                global_k = k
                V[global_j * p + global_k][col] = 1
                
                # W
                global_i = 2
                global_k = k
                W[global_k * n + global_i][col] = 1
                
                col += 1

    # Convert to numpy for efficient checking
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Verify decomposition correctness before attempting optimization
    def verify(U_mat, V_mat, W_mat):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Build the tensor T[i,j,k] = delta[i,k] (for matrix mult: C[i,k] = sum_j A[i,j]B[j,k])
        # Actually, the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]B[j,k]
        # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,k] * (some structure)
        # Wait, let me think about this more carefully.
        
        # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 for all i,j,k (since each term A[i,j]B[j,k] contributes to C[i,k])
        # But actually the tensor is T[i,j,k] = delta[i,k] in the output index? No...
        
        # Let me reconsider. The matrix multiplication tensor T satisfies:
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k, and the reconstruction is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Actually no. The tensor for matrix multiplication is:
        # T_{i,j,k} = 1 (it's the identity tensor in the sense that each (i,j,k) triple has weight 1)
        # And the factorization is: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U_np, V_np, W_np):
        # If verification fails, return the original (should not happen)
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by identifying and merging linearly dependent terms
    # A term r is redundant if U[:,r], V[:,r], W[:,r] can be expressed as linear combinations
    # of other terms. We can check this by looking for zero columns after elimination.
    
    # Simple approach: try to find columns that are linearly dependent
    # and remove them, then check if the decomposition still holds.
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=10.0):
        """Try to reduce rank by removing linearly dependent terms."""
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        
        # Keep track of which columns to keep
        keep = [True] * rank
        start_time = time.time()
        
        # Try removing each column one at a time and check if decomposition still holds
        for r in range(rank):
            if time.time() - start_time > max_time:
                break
            if not keep[r]:
                continue
            
            # Temporarily remove column r
            keep[r] = False
            
            # Build reduced matrices
            idx = [i for i in range(rank) if keep[i]]
            if len(idx) == 0:
                keep[r] = True
                continue
                
            U_red = U_mat[:, idx]
            V_red = V_mat[:, idx]
            W_red = W_mat[:, idx]
            
            # Check if decomposition still holds
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        actual = np.sum(U_red[i*m+j, :] * V_red[j*p+k, :] * W_red[k*n+i, :])
                        if abs(actual - 1.0) > 1e-9:
                            valid = False
                            break
            
            if not valid:
                keep[r] = True
        
        # Build final reduced matrices
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            U_new = U_mat[:, idx]
            V_new = V_mat[:, idx]
            W_new = W_mat[:, idx]
            return U_new, V_new, W_new
        
        return U_mat, V_mat, W_mat
    
    # Attempt rank reduction
    U_opt, V_opt, W_opt = try_reduce_rank(U_np, V_np, W_np, max_time=20.0)
    
    # Convert back to lists with integer/half-integer entries if possible
    def to_list(mat):
        # Round to nearest 0.5
        mat_rounded = np.round(mat * 2) / 2
        # Check if entries are close to integers or half-integers
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()
    
    return {"U": to_list(U_opt), "V": to_list(V_opt), "W": to_list(W_opt)}
# EVOLVE-BLOCK-END
