"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def check_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def check_decomp_fast(U, V, W, tol=1e-6):
        """Vectorized check for speed."""
        rank = U.shape[1]
        # Build the full tensor of expected values (all 1s) and computed values
        # This is O(n*m*p*rank) which is 64*rank. For rank~32, this is ~2000 ops.
        # But we can do it smarter with broadcasting if we construct the tensors.
        
        # Let's stick to the loop for correctness first, it's fast enough for rank < 100
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m+j
                    v_idx = j*p+k
                    w_idx = k*n+i
                    val = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Strategy:
    # 1. Pad A (2x4) to A_p (4x4) with zeros in rows 2,3.
    # 2. Pad B (4x8) to B_p (4x4) with zeros in cols 4,5,6,7? No, B is 4x8.
    #    Wait, the problem is 2x4 * 4x8.
    #    Let's view this as block multiplication.
    #    A = [A11, A12] where A11, A12 are 2x2.
    #    B = [B11; B21; B31; B41] where each Bi1 is 2x8? No, B is 4x8.
    #    B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8.
    #    C = A11 * B_top + A12 * B_bottom.
    #    C is 2x8.
    
    # Let's try the 4x4 padding idea from the directive.
    # Pad A (2x4) to 4x4:
    # A_p = [[A11, A12], [0, 0]]  (4x4)
    # Pad B (4x8) to 4x4? We can't pad a 4x8 matrix to 4x4 while preserving the product structure easily unless we split B.
    # The directive says: "pad the 2x4 matrix A to 4x4 ... and the 4x8 matrix B to 4x4 ... then apply rank-32 Strassen ... and project".
    # This implies we might need to split B into two 4x4 matrices?
    # B = [B_left, B_right] where B_left, B_right are 4x4.
    # Then C = A_p * B_left gives C_left (2x4), and A_p * B_right gives C_right (2x4).
    # Total C = [C_left, C_right] (2x8).
    # So we do two 4x4 multiplications? No, that would be rank 32 + 32 = 64.
    # Unless we can combine them?
    
    # Actually, let's look at the structure.
    # A = [A00, A01, A10, A11] (2x4)
    # B = [B00, B01, B02, B03; B10, B11, B12, B13; B20, B21, B22, B23; B30, B31, B32, B33] (4x8)
    # C = A * B (2x8)
    
    # Let's try a different structural decomposition.
    # We can view this as a 2x4 * 4x8 multiplication.
    # This is equivalent to two 2x2 * 2x8 multiplications?
    # C[:, 0:4] = A[:, 0:2] * B[0:2, 0:4] + A[:, 2:4] * B[2:4, 0:4]
    # C[:, 4:8] = A[:, 0:2] * B[0:2, 4:8] + A[:, 2:4] * B[2:4, 4:8]
    
    # Let's use the known optimal rank for 2x4 * 4x8.
    # The rank of matrix multiplication tensor M_{n,m,p} is at most n*m*p.
    # For 2x2*2x2, rank is 7 (Strassen).
    # For 2x4*4x8, we can decompose it into 8 blocks of 2x2*2x2?
    # A is 2x4, B is 4x8.
    # Split A into two 2x2 blocks: A_L, A_R.
    # Split B into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR. (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    # Wait, B is 4x8.
    # B = [ B_TL B_TR; B_BL B_BR ] where each block is 2x4.
    # So B_TL is 2x4, B_TR is 2x4, B_BL is 2x4, B_BR is 2x4.
    # C = A_L * B_TL + A_R * B_BL + A_L * B_TR + A_R * B_BR? No.
    # C = [A_L, A_R] * [B_TL; B_BL] + [A_L, A_R] * [B_TR; B_BR]
    # C_left (2x4) = A_L * B_TL + A_R * B_BL
    # C_right (2x4) = A_L * B_TR + A_R * B_BR
    
    # Each term like A_L * B_TL is a 2x2 * 2x4 multiplication.
    # We can decompose 2x2 * 2x4 into two 2x2 * 2x2 multiplications.
    # So A_L * B_TL = A_L * [B_TL1, B_TL2] = [A_L*B_TL1, A_L*B_TL2].
    # Each A_L*B_TLx is 2x2*2x2, rank 7.
    # So A_L * B_TL has rank 14.
    # Total rank for C_left = 14 (from A_L*B_TL) + 14 (from A_R*B_BL) = 28.
    # Total rank for C = 28 (left) + 28 (right) = 56.
    
    # Can we do better?
    # Maybe we can share some Strassen terms?
    # Or use a different decomposition.
    
    # Let's try to construct the rank-28 decomposition using 4 Strassen blocks of 2x2*2x2.
    # Wait, 4 blocks * 7 = 28.
    # How do we map 2x4 * 4x8 to 4 blocks of 2x2 * 2x2?
    
    # Let's split A into A1, A2 (each 2x2).
    # Let's split B into B1, B2, B3, B4 (each 2x2)? No, B is 4x8.
    # B has 4 rows and 8 columns.
    # We can split B into 4 blocks of 2x4?
    # Or 8 blocks of 2x2?
    
    # Let's define the blocks:
    # A = [A0, A1] (2x2 each)
    # B = [B00, B01, B02, B03; B10, B11, B12, B13; B20, B21, B22, B23; B30, B31, B32, B33]
    # C = A * B
    # C[:, 0:2] = A0 * B[:, 0:2] + A1 * B[:, 2:4]
    # C[:, 2:4] = A0 * B[:, 2:4] + A1 * B[:, 4:6]
    # C[:, 4:6] = A0 * B[:, 4:6] + A1 * B[:, 6:8]
    # C[:, 6:8] = A0 * B[:, 6:8] + A1 * B[:, 8:10] (out of bounds)
    
    # Let's be precise.
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # B is 4x8. Rows 0,1,2,3. Cols 0,1,2,3,4,5,6,7.
    # C is 8x2. Rows 0..7. Cols 0,1.
    # C[k, i] = sum_j A[i, j] * B[j, k]
    
    # Let's split A into A_L (cols 0,1) and A_R (cols 2,3).
    # A_L is 2x2. A_R is 2x2.
    # C[k, i] = A_L[i, 0]*B[0, k] + A_L[i, 1]*B[1, k] + A_R[i, 0]*B[2, k] + A_R[i, 1]*B[3, k]
    
    # Let's split B into B_T (rows 0,1) and B_B (rows 2,3).
    # B_T is 2x8. B_B is 2x8.
    # C[k, i] = A_L[i, :] * B_T[:, k] + A_R[i, :] * B_B[:, k]
    
    # Now split B_T into B_TL (cols 0,1,2,3) and B_TR (cols 4,5,6,7).
    # B_TL is 2x4. B_TR is 2x4.
    # Similarly for B_B.
    
    # This leads to 8 terms of 2x2 * 2x4.
    # Each 2x2 * 2x4 can be done with rank 14 (two 2x2*2x2 Strassen).
    # Total rank 8 * 14 = 112. Too high.
    
    # We need to be smarter.
    # Let's use the 4x4 padding idea.
    # Pad A to 4x4: A_p = [[A_L, A_R], [0, 0]]
    # Pad B to 4x4? We can't fit 4x8 into 4x4.
    # But we can split B into two 4x4 matrices: B_L (cols 0-3) and B_R (cols 4-7).
    # Then C_L = A_p * B_L (4x4 result, but we only need top 2 rows).
    # C_R = A_p * B_R (4x4 result, but we only need top 2 rows).
    # C = [C_L, C_R] (2x8).
    
    # Rank of A_p * B_L is at most 32 (Strassen 4x4).
    # Rank of A_p * B_R is at most 32.
    # Total rank 64.
    
    # But A_p has zeros in the bottom half.
    # A_p = [A11, A12; 0, 0]
    # B_L = [B11, B12; B21, B22] (4x4)
    # C_L = A_p * B_L = [A11*B11 + A12*B21, A11*B12 + A12*B22; 0, 0]
    # We only need the top half of C_L, which is A11*B11 + A12*B21 (2x2) and A11*B12 + A12*B22 (2x2).
    # This is exactly the same as the original multiplication for the left half.
    
    # So the 4x4 Strassen decomposition will compute 16 output elements, but 8 of them are zero (bottom half).
    # Can we eliminate the terms that only contribute to the zero elements?
    # Yes! In the Strassen decomposition of 4x4, some terms contribute only to the bottom rows of the result.
    # If we can identify which terms contribute only to the bottom rows, we can drop them.
    
    # Let's implement this.
    # 1. Build the standard Strassen 4x4 decomposition (rank 32).
    # 2. Map the 4x4 indices to the 2x4 * 4x8 indices.
    # 3. For each term in the Strassen decomposition, check if it contributes to any of the "valid" output elements (top 2 rows of C_L and C_R).
    # 4. If a term contributes only to invalid (zero) elements, drop it.
    
    def build_strassen_4x4():
        """
        Build the standard Strassen decomposition for 4x4 * 4x4 -> 4x4.
        Returns U, V, W matrices of shape (16, 32), (16, 32), (16, 32).
        Indices: U[i*4+j, r] corresponds to A[i,j].
                 V[i*4+j, r] corresponds to B[i,j].
                 W[i*4+j, r] corresponds to C[i,j].
        """
        U = np.zeros((16, 32))
        V = np.zeros((16, 32))
        W = np.zeros((16, 32))
        
        # Strassen 4x4 is built from 4 Strassen 2x2 blocks? No.
        # Standard Strassen 4x4 uses 2^ceil(log2(4)/log2(2)) = 4 levels?
        # Actually, Strassen 4x4 can be done with rank 32.
        # It is composed of 4 Strassen 2x2 multiplications? No, that would be 4*7=28.
        # Wait, 4x4 multiplication can be done with rank 32 using Strassen.
        # It is equivalent to 4 multiplications of 2x2 matrices? No.
        # Strassen 4x4: Split into 4 2x2 blocks.
        # A = [A11, A12; A21, A22]
        # B = [B11, B12; B21, B22]
        # C = A*B
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # We can use Strassen for each 2x2*2x2? That would be 4*7=28.
        # But the terms are not independent.
        # Actually, the rank of 4x4 matrix multiplication is known to be 32?
        # No, the rank of 4x4 matrix multiplication is at most 32 (Strassen), and it is known that rank is 32?
        # Wait, Strassen 2x2 is rank 7.
        # 4x4 can be done with 4 Strassen 2x2 blocks?
        # C11 = Strassen(A11, B11) + Strassen(A12, B21)? No, that's not how it works.
        
        # Let's use the standard Strassen 4x4 algorithm.
        # It uses 32 multiplications.
        # The decomposition is:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B21
        # M6 = (A12-A21)(B21+B22)
        # M7 = (A21-A22)(B11+B12)
        # This is for 2x2.
        
        # For 4x4, we recursively apply Strassen.
        # 4x4 = 4 * 2x2 blocks.
        # We need to compute C11, C12, C21, C22.
        # Each is a 2x2*2x2 multiplication.
        # So we need 4 * 7 = 28 multiplications?
        # But the inputs are shared.
        # Actually, the rank of 4x4 matrix multiplication is 32?
        # Let's check the literature.
        # Rank of 4x4 matrix multiplication is 32? No, it is known to be at most 32, and lower bound is 27?
        # Actually, Strassen 4x4 uses 32 scalar multiplications.
        
        # Let's implement the 4 Strassen 2x2 blocks for the 4 quadrants.
        # Block 1: A11 * B11 -> C11_part1
        # Block 2: A12 * B21 -> C11_part2
        # Block 3: A11 * B12 -> C12_part1
        # Block 4: A12 * B22 -> C12_part2
        # Block 5: A21 * B11 -> C21_part1
        # Block 6: A22 * B21 -> C21_part2
        # Block 7: A21 * B12 -> C22_part1
        # Block 8: A22 * B22 -> C22_part2
        
        # This is 8 Strassen 2x2 blocks, each rank 7. Total rank 56.
        # This is not optimal.
        
        # The correct Strassen 4x4 algorithm uses 32 multiplications.
        # It is based on the 7 Strassen formulas but applied recursively.
        # Let's use a known rank-32 decomposition for 4x4.
        
        # For now, let's use the rank-56 decomposition (8 Strassen 2x2 blocks) and then prune.
        # Or better, let's use the rank-28 decomposition for 2x4*4x8 directly.
        
        pass

    # Let's go back to the rank-28 decomposition for 2x4*4x8.
    # We established that it can be decomposed into 4 Strassen 2x2 blocks.
    # Let's build that.
    
    def build_strassen_28():
        U = np.zeros((8, 28))
        V = np.zeros((32, 28))
        W = np.zeros((16, 28))
        
        # A is 2x4. Indices i*4+j.
        # B is 4x8. Indices j*8+k.
        # C is 8x2. Indices k*2+i.
        
        # We split A into A_L (cols 0,1) and A_R (cols 2,3).
        # We split B into B_T (rows 0,1) and B_B (rows 2,3).
        # We split B_T into B_TL (cols 0,1) and B_TR (cols 2,3).
        # We split B_B into B_BL (cols 0,1) and B_BR (cols 2,3).
        
        # C[:, 0:2] = A_L * B_TL + A_R * B_BL
        # C[:, 2:4] = A_L * B_TR + A_R * B_BR
        
        # Wait, B is 4x8.
        # B_T is rows 0,1. Cols 0..7.
        # B_TL is cols 0,1. B_TR is cols 2,3.
        # But we need cols 0..7 for C.
        # C has 8 columns.
        # So we need 4 blocks for B_T and 4 blocks for B_B?
        # B_T0 (cols 0,1), B_T1 (cols 2,3), B_T2 (cols 4,5), B_T3 (cols 6,7).
        # B_B0 (cols 0,1), B_B1 (cols 2,3), B_B2 (cols 4,5), B_B3 (cols 6,7).
        
        # C[:, 0:2] = A_L * B_T0 + A_R * B_B0
        # C[:, 2:4] = A_L * B_T1 + A_R * B_B1
        # C[:, 4:6] = A_L * B_T2 + A_R * B_B2
        # C[:, 6:8] = A_L * B_T3 + A_R * B_B3
        
        # Each term is 2x2 * 2x2.
        # So we have 8 Strassen blocks.
        # Total rank 8 * 7 = 56.
        
        # Can we share terms?
        # The A_L and A_R are shared across all blocks.
        # The B blocks are distinct.
        
        # Let's implement the 56-rank decomposition first, then try to reduce.
        
        def add_strassen_block(base_col, A_col_offset, B_row_offset, B_col_offset, C_col_offset):
            """
            A is 2x4. A_col_offset is 0 or 2.
            B is 4x8. B_row_offset is 0 or 2. B_col_offset is 0,2,4,6.
            C is 8x2. C_col_offset is 0,2,4,6.
            """
            # A indices
            A00 = 0 * 4 + (A_col_offset + 0)
            A01 = 0 * 4 + (A_col_offset + 1)
            A10 = 1 * 4 + (A_col_offset + 0)
            A11 = 1 * 4 + (A_col_offset + 1)
            
            # B indices
            B00 = (B_row_offset + 0) * 8 + (B_col_offset + 0)
            B01 = (B_row_offset + 0) * 8 + (B_col_offset + 1)
            B10 = (B_row_offset + 1) * 8 + (B_col_offset + 0)
            B11 = (B_row_offset + 1) * 8 + (B_col_offset + 1)
            
            # C indices
            # C is 8x2. C[k, i].
            # We are computing C[C_col_offset:C_col_offset+2, :]
            # So k = C_col_offset, C_col_offset+1.
            C00 = (C_col_offset + 0) * 2 + 0
            C01 = (C_col_offset + 0) * 2 + 1
            C10 = (C_col_offset + 1) * 2 + 0
            C11 = (C_col_offset + 1) * 2 + 1
            
            # M1 = (A00+A11)(B00+B11)
            col = base_col
            U[A00, col] = 1
            U[A11, col] = 1
            V[B00, col] = 1
            V[B11, col] = 1
            W[C00, col] = 1
            W[C11, col] = 1
            
            # M2 = (A10+A11)B00
            col = base_col + 1
            U[A10, col] = 1
            U[A11, col] = 1
            V[B00, col] = 1
            W[C10, col] = 1
            W[C11, col] = -1
            
            # M3 = A00(B01-B11)
            col = base_col + 2
            U[A00, col] = 1
            V[B01, col] = 1
            V[B11, col] = -1
            W[C01, col] = 1
            W[C11, col] = 1
            
            # M4 = A11(B10-B00)
            col = base_col + 3
            U[A11, col] = 1
            V[B10, col] = 1
            V[B00, col] = -1
            W[C00, col] = 1
            W[C10, col] = 1
            
            # M5 = (A00+A10)B10
            col = base_col + 4
            U[A00, col] = 1
            U[A10, col] = 1
            V[B10, col] = 1
            W[C00, col] = -1
            W[C01, col] = 1
            
            # M6 = (A01-A10)(B10+B11)
            col = base_col + 5
            U[A01, col] = 1
            U[A10, col] = -1
            V[B10, col] = 1
            V[B11, col] = 1
            W[C11, col] = 1
            
            # M7 = (A10-A11)(B00+B01)
            col = base_col + 6
            U[A10, col] = 1
            U[A11, col] = -1
            V[B00, col] = 1
            V[B01, col] = 1
            W[C00, col] = 1
            
            return base_col + 7
        
        # Block 1: A_L * B_T0 -> C[:, 0:2]
        next_col = add_strassen_block(0, 0, 0, 0, 0)
        # Block 2: A_R * B_B0 -> C[:, 0:2]
        next_col = add_strassen_block(next_col, 2, 2, 0, 0)
        
        # Block 3: A_L * B_T1 -> C[:, 2:4]
        next_col = add_strassen_block(next_col, 0, 0, 2, 2)
        # Block 4: A_R * B_B1 -> C[:, 2:4]
        next_col = add_strassen_block(next_col, 2, 2, 2, 2)
        
        # Block 5: A_L * B_T2 -> C[:, 4:6]
        next_col = add_strassen_block(next_col, 0, 0, 4, 4)
        # Block 6: A_R * B_B2 -> C[:, 4:6]
        next_col = add_strassen_block(next_col, 2, 2, 4, 4)
        
        # Block 7: A_L * B_T3 -> C[:, 6:8]
        next_col = add_strassen_block(next_col, 0, 0, 6, 6)
        # Block 8: A_R * B_B3 -> C[:, 6:8]
        next_col = add_strassen_block(next_col, 2, 2, 6, 6)
        
        return U, V, W

    U56, V56, W56 = build_strassen_28()
    
    if check_decomp(U56, V56, W56):
        U, V, W = U56, V56, W56
        current_rank = 56
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        current_rank = 64

    # Now try to reduce rank by finding linear dependencies.
    def try_reduce_rank(U, V, W):
        rank = U.shape[1]
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        
        for iteration in range(100):
            if time.time() > deadline - 10.0:
                break
            
            # Try to find a column that is a linear combination of others
            # in the tensor sense.
            best_elimination = None
            best_residual = float('inf')
            
            for r in range(rank):
                if rank <= 20:
                    break
                
                mask = np.ones(rank, dtype=bool)
                mask[r] = False
                
                # Sample some (i,j,k) constraints
                all_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                sampled = random.sample(all_indices, min(50, len(all_indices)))
                
                A_mat = np.zeros((len(sampled), rank - 1))
                b_vec = np.zeros(len(sampled))
                
                for idx, (i, j, k) in enumerate(sampled):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    for s, col_idx in enumerate(np.where(mask)[0]):
                        A_mat[idx, s] = U_curr[u_idx, col_idx] * V_curr[v_idx, col_idx] * W_curr[w_idx, col_idx]
                    
                    b_vec[idx] = U_curr[u_idx, r] * V_curr[v_idx, r] * W_curr[w_idx, r]
                
                try:
                    c_sol, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    residual = np.linalg.norm(A_mat @ c_sol - b_vec)
                    
                    if residual < 1e-6 and residual < best_residual:
                        best_residual = residual
                        best_elimination = (r, c_sol, np.where(mask)[0])
                except:
                    pass
            
            if best_elimination is not None:
                r, c_sol, cols = best_elimination
                
                U_test = np.delete(U_curr, r, axis=1)
                V_test = np.delete(V_curr, r, axis=1)
                W_test = np.delete(W_curr, r, axis=1)
                
                if check_decomp(U_test, V_test, W_test, tol=1e-5):
                    U_curr, V_curr, W_curr = U_test, V_test, W_test
                    rank -= 1
            
            if rank <= 20:
                break
        
        return U_curr, V_curr, W_curr

    U_new, V_new, W_new = try_reduce_rank(U, V, W)
    
    if check_decomp(U_new, V_new, W_new) and U_new.shape[1] < current_rank:
        U, V, W = U_new, V_new, W_new
        current_rank = U_new.shape[1]
    
    if not check_decomp(U, V, W):
        if check_decomp(U56, V56, W56):
            U, V, W = U56, V56, W56
        else:
            rank = n * m * p
            U = np.zeros((U_dim, rank))
            V = np.zeros((V_dim, rank))
            W = np.zeros((W_dim, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
