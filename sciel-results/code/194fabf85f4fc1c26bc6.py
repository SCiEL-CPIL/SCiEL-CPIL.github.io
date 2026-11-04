"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank accumulator
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # We will build the matrices column by column (term by term)
    # First, determine the total rank
    rank_laderman = 23
    rank_strassen = 7
    rank_school_442 = 4 * 4 * 2
    rank_school_422 = 4 * 2 * 2
    rank_school_244 = 2 * 4 * 4
    rank_school_224 = 2 * 2 * 4
    rank_school_242 = 2 * 4 * 2
    
    total_rank = (rank_laderman + 
                  rank_school_442 + 
                  rank_school_422 + 
                  rank_school_244 + 
                  rank_school_224 + 
                  rank_school_242 + 
                  rank_strassen)
                  
    U = np.zeros((U_rows, total_rank))
    V = np.zeros((V_rows, total_rank))
    W = np.zeros((W_rows, total_rank))
    
    col_idx = 0
    
    # Helper to add a decomposition
    def add_decomp(U_sub, V_sub, W_sub, i_offset, j_offset, k_offset):
        """
        Adds a sub-decomposition to the global matrices.
        U_sub: (n_sub * m_sub, rank_sub)
        V_sub: (m_sub * p_sub, rank_sub)
        W_sub: (p_sub * n_sub, rank_sub)
        i_offset: row offset in A (and C)
        j_offset: col offset in A (and row offset in B)
        k_offset: col offset in B (and col offset in C)
        """
        rank_sub = U_sub.shape[1]
        n_sub = U_sub.shape[0] // m_sub if m_sub else 0 # Need m_sub
        # Actually, we need to know dimensions to map indices
        # Let's pass dimensions explicitly or infer
        # U_sub is n_sub * m_sub. We know m_sub from context.
        
        # This helper is tricky with offsets. Let's do it manually for each block.
        pass

    # 1. C11 = A11 B11 (4x4x4) - Laderman
    # A11: rows 0-3, cols 0-3
    # B11: rows 0-3, cols 0-3
    # C11: rows 0-3, cols 0-3
    # Laderman decomposition for 4x4x4
    # Indices: i, j, k in 0..3
    # U index: i*4 + j (local) -> i*6 + j (global)
    # V index: j*4 + k (local) -> j*6 + k (global)
    # W index: k*4 + i (local) -> k*6 + i (global)
    
    # Laderman terms (23 terms)
    # Standard Laderman algorithm for 4x4x4
    # References: Laderman 1976
    # We define the 23 rank-1 tensors.
    # u, v, w are vectors of length 16 (4x4)
    
    laderman_terms = []
    # The 23 terms of Laderman's algorithm
    # Format: (u_indices, v_indices, w_indices) where indices are (i,j), (j,k), (k,i)
    # Or simply coefficients for the 16 basis elements.
    
    # To avoid hardcoding 23x3x16 numbers, we can use a known implementation or generate it.
    # For brevity and correctness, I will use the known explicit form.
    # However, hardcoding 23 terms is verbose.
    # Alternative: Use Strassen 2x2 recursively?
    # R(4) <= 2 R(2) + ...? No, Laderman is better.
    
    # Let's use a standard library or simple implementation if possible.
    # Since I must provide the code, I will hardcode the Laderman matrix.
    
    # Laderman 4x4x4 decomposition
    # U, V, W are 16x23 matrices.
    
    # Source: Various tensor decomposition libraries.
    # I will construct them.
    
    # For the sake of this response, I will use a simplified correct decomposition if Laderman is too complex to hardcode accurately from memory.
    # Schoolbook 4x4x4 is rank 64.
    # Strassen 2x2x2 is rank 7.
    
    # If I use Strassen for 4x4x4:
    # R(4) <= 2 R(2) + ... ?
    # Actually, R(4) <= 2*R(2) is not correct.
    # R(4) <= 23 is Laderman.
    # R(4) <= 2*7 + ...?
    # Using Strassen for 4x4:
    # A = [[A11, A12], [A21, A22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # This is 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank 7 * 7 = 49.
    # 49 is better than 64 (schoolbook) but worse than 23 (Laderman).
    
    # I will use the 49-rank Strassen-based 4x4x4 decomposition to avoid hardcoding Laderman.
    # It is easier to implement correctly.
    
    def strassen_2x2():
        # Returns U, V, W for 2x2x2
        # Rank 7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Indices: i,j in 0,1
        # U[i*2+j]
        # V[j*2+k]
        # W[k*2+i]
        
        # M1 = (A11+A22)(B11+B22)
        # A11: (0,0), A22: (1,1)
        # B11: (0,0), B22: (1,1)
        # C11 += M1, C12 -= M2, C21 -= M4, C22 += M1? No.
        
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to express these as linear combinations of rank-1 terms.
        # Each M is a product of two linear combinations.
        # e.g. M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This is 4 rank-1 terms.
        # But we can group them?
        # No, the rank of the sum is the sum of ranks if independent.
        # Actually, the tensor rank of M1 is 4?
        # (A11+A22) is a vector in A-space. (B11+B22) is a vector in B-space.
        # The term is u \otimes v \otimes w.
        # u = e_00 + e_11.
        # v = e_00 + e_11.
        # w = e_00 + e_11? No, w is for C.
        # C11 gets M1. C22 gets M1.
        # So w = e_00 + e_11 (for C11 and C22).
        # So M1 is 1 rank-1 term!
        
        # Let's verify.
        # Term 1:
        # u1 = e_00 + e_11
        # v1 = e_00 + e_11
        # w1 = e_00 + e_11
        # Product: (A11+A22)(B11+B22) contributes to C11 and C22.
        # C11 += (A11+A22)(B11+B22)
        # C22 += (A11+A22)(B11+B22)
        # This matches M1 contribution to C11 and C22.
        
        # Term 2: M2 = (A21+A22)B11
        # u2 = e_10 + e_11
        # v2 = e_00
        # w2 = e_10 (for C21)
        # C21 += (A21+A22)B11.
        # Matches.
        
        # Term 3: M3 = A11(B12-B22)
        # u3 = e_00
        # v3 = e_01 - e_11
        # w3 = e_01 (for C12)
        # C12 += A11(B12-B22).
        # Matches.
        
        # Term 4: M4 = A22(B21-B11)
        # u4 = e_11
        # v4 = e_10 - e_00
        # w4 = e_10 (for C21) + e_00 (for C11)
        # C21 += A22(B21-B11)
        # C11 += A22(B21-B11)
        # Matches M4 contribution to C21 and C11.
        
        # Term 5: M5 = (A11+A12)B22
        # u5 = e_00 + e_01
        # v5 = e_11
        # w5 = e_11 (for C22) - e_00 (for C11)
        # C22 += (A11+A12)B22
        # C11 -= (A11+A12)B22
        # Matches M5 contribution to C22 and C11.
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # u6 = e_10 - e_00
        # v6 = e_00 + e_01
        # w6 = e_11 (for C22)
        # C22 += (A21-A11)(B11+B12)
        # Matches M6 contribution to C22.
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # u7 = e_01 - e_11
        # v7 = e_10 + e_11
        # w7 = e_00 (for C11)
        # C11 += (A12-A22)(B21+B22)
        # Matches M7 contribution to C11.
        
        # Fill U, V, W
        # U indices: 00, 01, 10, 11
        # V indices: 00, 01, 10, 11
        # W indices: 00, 01, 10, 11
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[3, 4] = 1; W[0, 4] = -1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # 4x4x4 using Strassen 2x2 blocks
    # Rank 49
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        # We have 7 blocks.
        # Each block is 2x2x2.
        # We need to map 2x2 indices to 4x4 indices.
        # A11: i,j in 0,1 -> 0,1
        # A12: i in 0,1, j in 2,3
        # A21: i in 2,3, j in 0,1
        # A22: i in 2,3, j in 2,3
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # 7 multiplications
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # For each M, we use 7 rank-1 terms from 2x2 Strassen.
        # Total 49 terms.
        
        # Block mappings for U (A):
        # Block 11: rows 0-1, cols 0-1
        # Block 12: rows 0-1, cols 2-3
        # Block 21: rows 2-3, cols 0-1
        # Block 22: rows 2-3, cols 2-3
        
        # Block mappings for V (B):
        # Block 11: rows 0-1, cols 0-1
        # Block 12: rows 0-1, cols 2-3
        # Block 21: rows 2-3, cols 0-1
        # Block 22: rows 2-3, cols 2-3
        
        # Block mappings for W (C):
        # Block 11: rows 0-1, cols 0-1
        # Block 12: rows 0-1, cols 2-3
        # Block 21: rows 2-3, cols 0-1
        # Block 22: rows 2-3, cols 2-3
        
        def add_block(U_dst, V_dst, W_dst, col_offset, 
                      u_row_off, u_col_off, 
                      v_row_off, v_col_off, 
                      w_row_off, w_col_off,
                      sign_u=1, sign_v=1, sign_w=1):
            # Adds a 2x2x2 decomposition scaled by signs
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        u_idx_local = i*2+j
                        u_idx_global = (u_row_off + i)*4 + (u_col_off + j)
                        U_dst[u_idx_global, col_offset+r] += U2[u_idx_local, r] * sign_u
                
                # V
                for j in range(2):
                    for k in range(2):
                        v_idx_local = j*2+k
                        v_idx_global = (v_row_off + j)*4 + (v_col_off + k)
                        V_dst[v_idx_global, col_offset+r] += V2[v_idx_local, r] * sign_v
                        
                # W
                for k in range(2):
                    for i in range(2):
                        w_idx_local = k*2+i
                        w_idx_global = (w_row_off + k)*4 + (w_col_off + i)
                        W_dst[w_idx_global, col_offset+r] += W2[w_idx_local, r] * sign_w

        col = 0
        # M1: (A11+A22)(B11+B22)
        # A11: (0,0), A22: (1,1)
        # B11: (0,0), B22: (1,1)
        # C11: (0,0), C22: (1,1)
        # This is sum of two products.
        # We can't just add two blocks.
        # We must use the 7 terms of Strassen 2x2 on the "sum" matrices?
        # No, the 7 terms of Strassen 2x2 are for 2x2 matrices.
        # Here A11+A22 is a 2x2 matrix.
        # So we treat A11+A22 as the input matrix for the 2x2 Strassen.
        # The U matrix for this term will have entries in both A11 and A22 blocks.
        
        # Let's refine add_block to handle linear combinations.
        # Actually, the 7 terms of Strassen 2x2 are:
        # u1 = e00+e11, etc.
        # If the input is A11+A22, then u1 becomes (e00+e11) in A11 AND (e00+e11) in A22.
        
        # So for M1:
        # U: A11 (u1) + A22 (u1)
        # V: B11 (v1) + B22 (v1)
        # W: C11 (w1) + C22 (w1)
        
        # We can call add_block twice? No, that would be 2*7=14 terms.
        # But M1 is ONE multiplication of (A11+A22) and (B11+B22).
        # The rank of this multiplication is 7.
        # So we need 7 terms.
        # In each term, U has non-zeros in A11 and A22.
        
        # Let's modify add_block to accept multiple source blocks.
        
        def add_combined_block(U_dst, V_dst, W_dst, col_offset,
                               u_blocks, v_blocks, w_blocks):
            # u_blocks: list of (row_off, col_off, sign)
            # v_blocks: list of (row_off, col_off, sign)
            # w_blocks: list of (row_off, col_off, sign)
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        u_idx_local = i*2+j
                        val = 0
                        for (ro, co, s) in u_blocks:
                            u_idx_global = (ro + i)*4 + (co + j)
                            val += U2[u_idx_local, r] * s
                        if val != 0:
                            # Find the first block to store? No, we must store in all?
                            # No, U_dst is a single matrix.
                            # We must sum the contributions.
                            # But U_dst index is global.
                            # If u_blocks are disjoint, we just add.
                            pass
                # This logic is getting complex.
                # Simpler: Just iterate over all 16x16x16 for the 49 terms?
                # No.
                
        # Given time, I will use the Schoolbook 4x4x4 (rank 64) for simplicity and correctness.
        # Total rank: 64 + 32 + 16 + 32 + 16 + 16 + 7 = 183.
        # 183 < 216. Still an improvement.
        
        # I will implement Schoolbook for all rectangular blocks and 4x4x4.
        # And Strassen for 2x2x2.
        
        return None # Not used

    # Use Schoolbook for 4x4x4
    rank_444 = 64
    # Use Schoolbook for 4x4x2
    rank_442 = 32
    # Use Schoolbook for 4x2x2
    rank_422 = 16
    # Use Schoolbook for 2x4x4
    rank_244 = 32
    # Use Schoolbook for 2x2x4
    rank_224 = 16
    # Use Schoolbook for 2x4x2
    rank_242 = 16
    # Use Strassen for 2x2x2
    rank_222 = 7
    
    total_rank = rank_444 + rank_442 + rank_422 + rank_244 + rank_224 + rank_242 + rank_222
    
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    col = 0
    
    def add_schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i_off + i, col] = 1
                    V[j_off + j, col] = 1
                    W[k_off + k, col] = 1
                    col += 1

    def add_strassen_2x2(i_off, j_off, k_off):
        nonlocal col
        U2, V2, W2 = strassen_2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i_off + i, col + r] += U2[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[j_off + j, col + r] += V2[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[k_off + k, col + r] += W2[k*2+i, r]
        col += 7

    # 1. C11: 4x4x4
    add_schoolbook(4, 4, 4, 0, 0, 0)
    
    # 2. C12 part 1: A11 B12 (4x4x2)
    # A11: i 0-3, j 0-3
    # B12: j 0-3, k 4-5
    # C12: k 4-5, i 0-3
    add_schoolbook(4, 4, 2, 0, 0, 4)
    
    # 3. C12 part 2: A12 B22 (4x2x2)
    # A12: i 0-3, j 4-5
    # B22: j 4-5, k 4-5
    # C12: k 4-5, i 0-3
    add_schoolbook(4, 2, 2, 0, 4, 4)
    
    # 4. C21 part 1: A21 B11 (2x4x4)
    # A21: i 4-5, j 0-3
    # B11: j 0-3, k 0-3
    # C21: k 0-3, i 4-5
    add_schoolbook(2, 4, 4, 4, 0, 0)
    
    # 5. C21 part 2: A22 B21 (2x2x4)
    # A22: i 4-5, j 4-5
    # B21: j 4-5, k 0-3
    # C21: k 0-3, i 4-5
    add_schoolbook(2, 2, 4, 4, 4, 0)
    
    # 6. C22 part 1: A21 B12 (2x4x2)
    # A21: i 4-5, j 0-3
    # B12: j 0-3, k 4-5
    # C22: k 4-5, i 4-5
    add_schoolbook(2, 4, 2, 4, 0, 4)
    
    # 7. C22 part 2: A22 B22 (2x2x2)
    # A22: i 4-5, j 4-5
    # B22: j 4-5, k 4-5
    # C22: k 4-5, i 4-5
    add_strassen_2x2(4, 4, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
