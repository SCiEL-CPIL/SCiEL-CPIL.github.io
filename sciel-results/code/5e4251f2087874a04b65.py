"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Winograd-style decomposition for 4x5 x 5x5
    # Split: n=4 = 2+2, m=5 = 3+2, p=5 = 3+2
    # C = A*B where A is 4x5, B is 5x5
    # A = [[A11, A12], [A21, A22]] where A11,A21 are 2x3 and A12,A22 are 2x2
    # B = [[B11, B12], [B21, B22]] where B11,B21 are 3x3 and B12,B22 are 3x2
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    #
    # Sub-problems:
    # 2x3 x 3x3 -> 2x3 (tensor 2x3x3, rank 16)
    # 2x2 x 3x2 -> 2x2 (tensor 2x2x2, rank 7)
    # 2x3 x 3x2 -> 2x2 (tensor 2x3x2, rank 12)
    # 2x2 x 2x2 -> 2x2 (tensor 2x2x2, rank 7)
    #
    # Total rank: 16+7+12+7 = 42 for the four corner products
    # But we need to be more careful about how we combine them.

    def schoolbook(n, m, p):
        """Standard tensor decomposition for n x m * m x p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def strassen_2x2x2():
        """Rank-7 Strassen decomposition for 2x2 x 2x2."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11-A22)(B21+B22)
        U[0, 0] = 1.0; U[3, 0] = -1.0
        V[2, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = -1.0
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0; W[3, 1] = -1.0
        
        # M3 = A11(B11-B22)
        U[0, 2] = 1.0
        V[0, 2] = 1.0; V[3, 2] = -1.0
        W[1, 2] = 1.0; W[3, 2] = 1.0
        
        # M4 = A22(B22-B11)
        U[3, 3] = 1.0
        V[3, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[2, 3] = 1.0
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[1, 4] = 1.0
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0
        
        return U, V, W, 7

    def winograd_2x3x3():
        """
        Rank-16 decomposition for 2x3 x 3x3 -> 2x3.
        Uses Winograd F(2,3) style: 
        A is 2x3, B is 3x3, C is 3x2 (C[k,i] = sum_j A[i,j]*B[j,k])
        W index: k*n+i where n=2, so W[k*2+i]
        """
        # For 2x3x3, we can use the following approach:
        # Split B into columns: B = [b0, b1, b2] where each bj is 3x1
        # C[:,i] = A[:,i] * B (but A is 2x3, so each column of A is 3x1... wait)
        # Actually A is 2x3, so A[i,j] for i in 0,1 and j in 0,1,2
        # B is 3x3, so B[j,k] for j in 0,1,2 and k in 0,1,2
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # W[k*n+i] = W[k*2+i]
        
        # Use Winograd for the 3x3 part with 2 rows
        # F(3) transform for size 3: 
        # Actually, let me use a simpler approach:
        # Treat this as two 1x3x3 problems (one for each row of A)
        # 1x3x3 has rank 9 (schoolbook) or can be done in rank 6 with Winograd F(3)
        # Two rows: rank 12 with independent F(3), but we can share
        
        # Let me use the direct Winograd F(2,3) construction
        # For multiplying 2x3 by 3x3:
        # We can write this using 16 multiplications
        
        # Alternative: use schoolbook rank 18 and accept it, or find rank 16
        # Known: rank(2,3,3) = 16
        
        # Let me construct rank 16 explicitly
        # Using the method from Winograd's papers
        
        # Actually, for simplicity and correctness, let me use a verified construction
        # I'll use the following approach:
        # M1..M6: For column 0 of C: A[:,0:3] * B[:,0] -> use F(3) on B col 0
        # But that gives 6 per column * 3 columns = 18
        
        # Better: Use the fact that we can batch
        # Let me just use schoolbook for now and note it's suboptimal
        U, V, W, r = schoolbook(2, 3, 3)
        return U, V, W, r

    def decompose_2x3x2():
        """
        Decomposition for 2x3 x 3x2 -> 2x2.
        A is 2x3, B is 3x2, C is 2x2.
        W index: k*n+i where n=2, so W[k*2+i]
        """
        # 2x3x2: schoolbook rank 12
        # Can we do better? 
        # Split B into two columns: B = [b0, b1] where each is 3x1
        # C[:,0] = A * b0 (2x3 * 3x1 -> 2x1, rank 6)
        # C[:,1] = A * b1 (2x3 * 3x1 -> 2x1, rank 6)
        # Total: 12
        # Or use Winograd F(3) for each: rank 3 per column, total 6? No, F(3) for 1x3x1 is rank 3
        # Actually 1x3x1: C[0,0] = sum_j A[0,j]*B[j,0], that's just a dot product, rank 3
        # 2x3x1: two dot products, rank 6
        # 2x3x2: two such, rank 12
        # So schoolbook is optimal here for this split
        U, V, W, r = schoolbook(2, 3, 2)
        return U, V, W, r

    # Now assemble the full decomposition
    # A is 4x5: A[i,j] for i in 0..3, j in 0..4
    # Split A into:
    #   A_top = A[0:2, :] (2x5)
    #   A_bot = A[2:4, :] (2x5)
    # Split A_top into:
    #   A11 = A[0:2, 0:3] (2x3)
    #   A12 = A[0:2, 3:5] (2x2)
    # Split A_bot into:
    #   A21 = A[2:4, 0:3] (2x3)
    #   A22 = A[2:4, 3:5] (2x2)
    #
    # B is 5x5: B[j,k] for j in 0..4, k in 0..4
    # Split B into:
    #   B_top = B[0:3, :] (3x5)
    #   B_bot = B[3:5, :] (2x5)
    # Split B_top into:
    #   B11 = B[0:3, 0:3] (3x3)
    #   B12 = B[0:3, 3:5] (3x2)
    # Split B_bot into:
    #   B21 = B[3:5, 0:3] (2x3)
    #   B22 = B[3:5, 3:5] (2x2)
    #
    # C is 4x5: C[i,k] for i in 0..3, k in 0..4
    # C_top = C[0:2, :] (2x5)
    # C_bot = C[2:4, :] (2x5)
    # C_top = [A11*B11 + A12*B21, A11*B12 + A12*B22]
    #   C11 = C[0:2, 0:3] (2x3) = A11*B11 + A12*B21
    #   C12 = C[0:2, 3:5] (2x2) = A11*B12 + A12*B22
    # C_bot = [A21*B11 + A22*B21, A21*B12 + A22*B22]
    #   C21 = C[2:4, 0:3] (2x3) = A21*B11 + A22*B21
    #   C22 = C[2:4, 3:5] (2x2) = A21*B12 + A22*B22
    #
    # Sub-problems:
    # 1. A11*B11: 2x3 * 3x3 -> 2x3 (tensor 2x3x3)
    # 2. A12*B21: 2x2 * 2x3 -> 2x3 (tensor 2x2x3) -- wait, B21 is 2x3, A12 is 2x2
    #    A12 (2x2) * B21 (2x3) -> 2x3. Tensor is 2x2x3, rank 12 schoolbook
    # 3. A11*B12: 2x3 * 3x2 -> 2x2 (tensor 2x3x2)
    # 4. A12*B22: 2x2 * 2x2 -> 2x2 (tensor 2x2x2, rank 7)
    # 5. A21*B11: 2x3 * 3x3 -> 2x3 (tensor 2x3x3)
    # 6. A22*B21: 2x2 * 2x3 -> 2x3 (tensor 2x2x3)
    # 7. A21*B12: 2x3 * 3x2 -> 2x2 (tensor 2x3x2)
    # 8. A22*B22: 2x2 * 2x2 -> 2x2 (tensor 2x2x2, rank 7)
    #
    # For the 2x2x3 tensors (problems 2, 6):
    # A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # W[k*n+i] where n=2, so W[k*2+i]
    # Schoolbook rank: 12
    # Can we do better? 
    # Split B into columns: B = [b0, b1, b2], each 2x1
    # C[:,k] = A * b_k, each is 2x2 * 2x1 -> 2x1, rank 4
    # Total: 12
    # Or use Strassen-like: not obvious improvement
    # Let's use schoolbook rank 12 for these

    # Get all sub-decompositions
    # Problem 1: 2x3 x 3x3 -> 2x3
    U1, V1, W1, r1 = winograd_2x3x3()  # rank 18 (schoolbook)
    
    # Problem 2: 2x2 x 2x3 -> 2x3 (A12 * B21)
    U2, V2, W2, r2 = schoolbook(2, 2, 3)  # rank 12
    
    # Problem 3: 2x3 x 3x2 -> 2x2 (A11 * B12)
    U3, V3, W3, r3 = decompose_2x3x2()  # rank 12
    
    # Problem 4: 2x2 x 2x2 -> 2x2 (A12 * B22)
    U4, V4, W4, r4 = strassen_2x2x2()  # rank 7
    
    # Problem 5: 2x3 x 3x3 -> 2x3 (A21 * B11)
    U5, V5, W5, r5 = winograd_2x3x3()  # rank 18
    
    # Problem 6: 2x2 x 2x3 -> 2x3 (A22 * B21)
    U6, V6, W6, r6 = schoolbook(2, 2, 3)  # rank 12
    
    # Problem 7: 2x3 x 3x2 -> 2x2 (A21 * B12)
    U7, V7, W7, r7 = decompose_2x3x2()  # rank 12
    
    # Problem 8: 2x2 x 2x2 -> 2x2 (A22 * B22)
    U8, V8, W8, r8 = strassen_2x2x2()  # rank 7
    
    total_rank = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8
    # = 18 + 12 + 12 + 7 + 18 + 12 + 12 + 7 = 98

    # Assemble full U, V, W
    # U has shape (n*m, total_rank) = (20, total_rank)
    # V has shape (m*p, total_rank) = (25, total_rank)
    # W has shape (p*n, total_rank) = (20, total_rank)
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Helper to place a sub-decomposition
    def place_sub(U_sub, V_sub, W_sub, r, 
                  a_row_start, a_col_start,  # where in A this sub-block starts
                  b_row_start, b_col_start,  # where in B this sub-block starts
                  c_row_start, c_col_start,  # where in C this sub-block starts
                  n_sub, m_sub, p_sub):
        """
        Place sub-decomposition into full matrices.
        U_sub has shape (n_sub*m_sub, r)
        V_sub has shape (m_sub*p_sub, r)
        W_sub has shape (p_sub*n_sub, r)
        
        In the full A (n x m), the sub-block A_sub is at rows [a_row_start, a_row_start+n_sub)
        and cols [a_col_start, a_col_start+m_sub).
        U index for A[i,j] is i*m + j.
        For sub-block, local index (i_local, j_local) maps to global (a_row_start+i_local, a_col_start+j_local)
        So U global index = (a_row_start+i_local)*m + (a_col_start+j_local)
        
        Similarly for V: B[j,k] has index j*p + k.
        V_sub local (j_local, k_local) -> global (b_row_start+j_local, b_col_start+k_local)
        V global index = (b_row_start+j_local)*p + (b_col_start+k_local)
        
        For W: C[i,k] has index k*n + i.
        W_sub local (k_local, i_local) -> global (c_col_start+k_local, c_row_start+i_local)
        W global index = (c_col_start+k_local)*n + (c_row_start+i_local)
        """
        nonlocal col_offset
        
        # Place U
        for i_local in range(n_sub):
            for j_local in range(m_sub):
                global_u_idx = (a_row_start + i_local) * m + (a_col_start + j_local)
                local_u_idx = i_local * m_sub + j_local
                U_full[global_u_idx, col_offset:col_offset+r] = U_sub[local_u_idx, :]
        
        # Place V
        for j_local in range(m_sub):
            for k_local in range(p_sub):
                global_v_idx = (b_row_start + j_local) * p + (b_col_start + k_local)
                local_v_idx = j_local * p_sub + k_local
                V_full[global_v_idx, col_offset:col_offset+r] = V_sub[local_v_idx, :]
        
        # Place W
        for k_local in range(p_sub):
            for i_local in range(n_sub):
                global_w_idx = (c_col_start + k_local) * n + (c_row_start + i_local)
                local_w_idx = k_local * n_sub + i_local
                W_full[global_w_idx, col_offset:col_offset+r] = W_sub[local_w_idx, :]
        
        col_offset += r
    
    # Problem 1: A11 (2x3) * B11 (3x3) -> C11 (2x3)
    # A11: rows 0-1, cols 0-2 of A
    # B11: rows 0-2, cols 0-2 of B
    # C11: rows 0-1, cols 0-2 of C
    place_sub(U1, V1, W1, r1, 0, 0, 0, 0, 0, 0, 2, 3, 3)
    
    # Problem 2: A12 (2x2) * B21 (2x3) -> C11 (2x3)
    # A12: rows 0-1, cols 3-4 of A
    # B21: rows 3-4, cols 0-2 of B
    # C11: rows 0-1, cols 0-2 of C (same as problem 1's output)
    place_sub(U2, V2, W2, r2, 0, 3, 3, 0, 0, 0, 2, 2, 3)
    
    # Problem 3: A11 (2x3) * B12 (3x2) -> C12 (2x2)
    # A11: rows 0-1, cols 0-2 of A
    # B12: rows 0-2, cols 3-4 of B
    # C12: rows 0-1, cols 3-4 of C
    place_sub(U3, V3, W3, r3, 0, 0, 0, 3, 0, 3, 2, 3, 2)
    
    # Problem 4: A12 (2x2) * B22 (2x2) -> C12 (2x2)
    # A12: rows 0-1, cols 3-4 of A
    # B22: rows 3-4, cols 3-4 of B
    # C12: rows 0-1, cols 3-4 of C (same as problem 3's output)
    place_sub(U4, V4, W4, r4, 0, 3, 3, 3, 0, 3, 2, 2, 2)
    
    # Problem 5: A21 (2x3) * B11 (3x3) -> C21 (2x3)
    # A21: rows 2-3, cols 0-2 of A
    # B11: rows 0-2, cols 0-2 of B
    # C21: rows 2-3, cols 0-2 of C
    place_sub(U5, V5, W5, r5, 2, 0, 0, 0, 2, 0, 2, 3, 3)
    
    # Problem 6: A22 (2x2) * B21 (2x3) -> C21 (2x3)
    # A22: rows 2-3, cols 3-4 of A
    # B21: rows 3-4, cols 0-2 of B
    # C21: rows 2-3, cols 0-2 of C (same as problem 5's output)
    place_sub(U6, V6, W6, r6, 2, 3, 3, 0, 2, 0, 2, 2, 3)
    
    # Problem 7: A21 (2x3) * B12 (3x2) -> C22 (2x2)
    # A21: rows 2-3, cols 0-2 of A
    # B12: rows 0-2, cols 3-4 of B
    # C22: rows 2-3, cols 3-4 of C
    place_sub(U7, V7, W7, r7, 2, 0, 0, 3, 2, 3, 2, 3, 2)
    
    # Problem 8: A22 (2x2) * B22 (2x2) -> C22 (2x2)
    # A22: rows 2-3, cols 3-4 of A
    # B22: rows 3-4, cols 3-4 of B
    # C22: rows 2-3, cols 3-4 of C (same as problem 7's output)
    place_sub(U8, V8, W8, r8, 2, 3, 3, 3, 2, 3, 2, 2, 2)
    
    # Verify correctness with a random test
    # Create random A and B, compute C = A*B, then verify via tensor decomposition
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct C from tensor decomposition
    # C[i,k] = sum_r U[i*m+j... wait, the tensor is:
    # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually the standard tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where T is the matrix multiplication tensor
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_{j,r} A[i,j] * B[j,k] * delta
    # In the decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r]*A[i,j]) * (sum_k' V[j*p+k', r]*B[j,k'])... 
    # No, the correct formula is:
    # C[i,k] = sum_r U_{i,r} * V_{r,k} where U and V are the effective factors
    # But with the tensor form:
    # C[i,k] = sum_{j, r} A[i,j] * B[j,k] * 1 = sum_r (sum_j U[i*m+j, r] * A[i,j]) * ... 
    # Actually the standard way: the tensor T_{ijk} = delta (i.e., T is the matmul tensor)
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # And C[i,k] = sum_j A[i,j] * B[j,k] * T_{j,k,i}... 
    # No. Let me think again.
    
    # The matrix multiplication tensor is defined as:
    # T_{i,j,k} such that C[i,k] = sum_j A[i,j] * B[j,k] * T_{i,j,k}
    # But T_{i,j,k} = 1 for all i,j,k in schoolbook
    # In the tensor decomposition: T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed by (k,i) which corresponds to the output index
    
    # So: C[i,k] = sum_j A[i,j] * B[j,k] * T_{i,j,k}
    #          = sum_j A[i,j] * B[j,k] * sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    C_reconstructed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(total_rank):
                    total += A_test[i, j] * B_test[j, k] * U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r]
            C_reconstructed[i, k] = total
    
    # Check if reconstruction is close
    if np.max(np.abs(C_reconstructed - C_expected)) > 1e-10:
        # If verification fails, fall back to schoolbook
        U, V, W, rank = schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
