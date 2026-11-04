"""Program for matrix multiplication 4x4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    # Schoolbook rank-128 decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build a rank-49 decomposition for 4x4 * 4x4 using Kronecker product of two Strassen levels
    # Then use it twice for 4x4 * 4x8 (left and right halves), giving rank 98.

    # Strassen 2x2 * 2x2 -> 2x2: rank 7
    # The 7 terms for Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5

    def strassen_2x2_decomposition():
        """
        Returns U, V, W for 2x2 * 2x2 -> 2x2 with rank 7.
        U shape: (2*2, 7) = (4, 7)  -- A entries indexed by (i,j) -> i*2+j
        V shape: (2*2, 7) = (4, 7)  -- B entries indexed by (j,k) -> j*2+k
        W shape: (2*2, 7) = (4, 7)  -- C entries indexed by (k,i) -> k*2+i
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))

        # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

        # M1 = (A00+A11)*(B00+B11)
        # A factor: A00 + A11
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        # B factor: B00 + B11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        # C: contributes to C00, C11
        W[0, 0] = 1   # C00
        W[3, 0] = 1   # C11

        # M2 = (A10+A11)*B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10
        W[3, 1] = 1  # C11

        # M3 = A00*(B01-B11)
        U[0, 2] = 1  # A00
        V[1, 2] = 1  # B01
        V[3, 2] = -1  # B11
        W[1, 2] = 1  # C01

        # M4 = A11*(B10-B00)
        U[3, 3] = 1  # A11
        V[2, 3] = 1  # B10
        V[0, 3] = -1  # B00
        W[0, 3] = -1  # C00 (negative because -M4 in C00)
        W[2, 3] = 1  # C10

        # M5 = (A00+A01)*B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[1, 4] = 1  # C01
        W[3, 4] = -1  # C11 (negative because -M5 in C11)

        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1  # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        W[0, 5] = 1  # C00

        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1  # A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        W[0, 6] = -1  # C00 (negative because -M7 in C00)

        return U, V, W

    def build_4x4x4():
        """
        Build rank-49 decomposition for 4x4 * 4x4 -> 4x4.
        Uses Kronecker product of two Strassen levels.
        
        A is 4x4, split into 2x2 blocks: A = [[A00, A01], [A10, A11]]
        B is 4x4, split into 2x2 blocks: B = [[B00, B01], [B10, B11]]
        C is 4x4, split into 2x2 blocks: C = [[C00, C01], [C10, C11]]
        
        At the top level, Strassen gives 7 "super-multiplications" of 2x2 blocks.
        Each 2x2 block multiplication is itself done with Strassen (rank 7).
        Total: 7 * 7 = 49 scalar multiplications.
        """
        # Get the base Strassen decomposition for 2x2*2x2
        U_base, V_base, W_base = strassen_2x2_decomposition()
        # U_base: (4, 7), V_base: (4, 7), W_base: (4, 7)
        
        # Now we need to lift this to 4x4*4x4.
        # The idea: the top-level Strassen operates on 2x2 blocks.
        # Each "block multiplication" (e.g., (A00+A11)*(B00+B11)) is a 2x2*2x2 product.
        # We use the base Strassen for each such product.
        
        # The top-level Strassen has 7 terms. For term t:
        #   A_factor_t = linear combination of A-blocks (each 2x2)
        #   B_factor_t = linear combination of B-blocks (each 2x2)
        #   The product A_factor_t * B_factor_t contributes to C-blocks with certain signs.
        
        # Let's define the top-level Strassen terms explicitly.
        # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1) in block space
        # B blocks: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1) in block space
        # C blocks: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1) in block space
        
        # Top-level Strassen terms (at block level):
        # T1: (A00+A11)*(B00+B11) -> contributes to C00 (+), C11 (+)
        # T2: (A10+A11)*B00 -> contributes to C10 (+), C11 (+)
        # T3: A00*(B01-B11) -> contributes to C01 (+)
        # T4: A11*(B10-B00) -> contributes to C00 (-), C10 (+)
        # T5: (A00+A01)*B11 -> contributes to C01 (+), C11 (-)
        # T6: (A10-A00)*(B00+B01) -> contributes to C00 (+)
        # T7: (A01-A11)*(B10+B11) -> contributes to C00 (-)
        
        # For each top-level term t, we need to express:
        #   A_factor_t = sum over block positions (bi, bj) of coeff_a * A_block(bi,bj)
        #   B_factor_t = sum over block positions (bk, bl) of coeff_b * B_block(bk,bl)
        #   The 2x2 product A_factor_t * B_factor_t is computed with base Strassen.
        #   The result contributes to C-blocks with signs.
        
        # Actually, let me think more carefully.
        # A_factor_t is a 2x2 matrix (linear combination of A-blocks).
        # B_factor_t is a 2x2 matrix (linear combination of B-blocks).
        # Their product P_t = A_factor_t * B_factor_t is a 2x2 matrix.
        # P_t contributes to C with: C = sum_t sign_t * P_t placed in appropriate block positions.
        
        # To compute P_t using base Strassen, we need:
        #   U_t = A_factor_t (as a 4-vector of A entries)
        #   V_t = B_factor_t (as a 4-vector of B entries)
        #   W_t = where P_t lands in C (as a 4-vector of C entries)
        
        # But A_factor_t itself is a linear combination of A-blocks, each of which is 2x2.
        # So A_factor_t has 4 entries, each being a linear combination of 4 A-entries.
        # Total: 16 A-entries involved, but with coefficients.
        
        # Let me define the top-level terms with their A and B block coefficients.
        
        # Block positions: (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
        # A_block(r, c) has entries A[r*2+i, c*2+j] for i,j in {0,1}
        # B_block(r, c) has entries B[r*2+i, c*2+j] for i,j in {0,1}
        # C_block(r, c) has entries C[r*2+i, c*2+j] for i,j in {0,1}
        
        # Top-level terms:
        # T1: A_factor = A00 + A11, B_factor = B00 + B11, C_contrib: C00 += P, C11 += P
        # T2: A_factor = A10 + A11, B_factor = B00, C_contrib: C10 += P, C11 += P
        # T3: A_factor = A00, B_factor = B01 - B11, C_contrib: C01 += P
        # T4: A_factor = A11, B_factor = B10 - B00, C_contrib: C00 -= P, C10 += P
        # T5: A_factor = A00 + A01, B_factor = B11, C_contrib: C01 += P, C11 -= P
        # T6: A_factor = A10 - A00, B_factor = B00 + B01, C_contrib: C00 += P
        # T7: A_factor = A01 - A11, B_factor = B10 + B11, C_contrib: C00 -= P
        
        # For each term t, A_factor_t is a 2x2 matrix. Let's write it as a 4-vector (flattened).
        # A_factor_t = [a00, a01, a10, a11] where each aij is a linear combo of A-entries.
        
        # Let me define the A and B coefficients for each top-level term.
        # A coefficient: for each A-entry (i, j) in 4x4, what's the coefficient in A_factor_t?
        # B coefficient: for each B-entry (j, k) in 4x4, what's the coefficient in B_factor_t?
        # W coefficient: for each C-entry (k, i) in 4x4, what's the coefficient?
        
        # Actually, it's cleaner to think of it as:
        # For term t, we have a 2x2 "A matrix" and a 2x2 "B matrix" (at block level).
        # The product is computed with base Strassen, giving 7 scalar multiplications.
        # Each scalar multiplication involves:
        #   - A linear combo of the 4 entries of the A matrix (which are themselves linear combos of A-entries)
        #   - A linear combo of the 4 entries of the B matrix (which are themselves linear combos of B-entries)
        #   - A linear combo of the 4 entries of the resulting 2x2 product (which map to C-entries)
        
        # Let me just build it directly.
        
        # For each top-level term t (0..6):
        #   Define A_blocks_t: dict from block_pos to coefficient
        #   Define B_blocks_t: dict from block_pos to coefficient
        #   Define C_blocks_t: dict from block_pos to coefficient (sign)
        
        top_terms = [
            # (A_blocks, B_blocks, C_blocks)
            # T1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # T2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): 1}),
            # T3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
            # T4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
            # T5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): -1}),
            # T6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
            # T7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1}),
        ]
        
        rank = 49
        U = np.zeros((n * m, rank))  # (16, 49)
        V = np.zeros((m * p, rank))  # (16, 49)
        W = np.zeros((p * n, rank))  # (16, 49)
        
        # Base Strassen: for a 2x2*2x2 product, we have 7 scalar multiplications.
        # U_base: (4, 7) -- A entries of the 2x2 matrix
        # V_base: (4, 7) -- B entries of the 2x2 matrix
        # W_base: (4, 7) -- C entries of the 2x2 result
        
        for t in range(7):
            A_blocks, B_blocks, C_blocks = top_terms[t]
            
            # For each base Strassen sub-term s (0..6):
            for s in range(7):
                col = t * 7 + s
                
                # A factor: U_base[:, s] gives coefficients for the 4 entries of the A 2x2 matrix.
                # The A 2x2 matrix has entries at positions (0,0), (0,1), (1,0), (1,1) in block space.
                # Each of these is a linear combination of A-entries based on A_blocks.
                
                # A 2x2 matrix entries:
                # A_mat[0,0] = sum over (br, bc) in A_blocks: A_blocks[(br,bc)] * A_block(br,bc)[0,0]
                # A_mat[0,1] = sum over (br, bc) in A_blocks: A_blocks[(br,bc)] * A_block(br,bc)[0,1]
                # A_mat[1,0] = sum over (br, bc) in A_blocks: A_blocks[(br,bc)] * A_block(br,bc)[1,0]
                # A_mat[1,1] = sum over (br, bc) in A_blocks: A_blocks[(br,bc)] * A_block(br,bc)[1,1]
                
                # A_block(br, bc)[i, j] = A[br*2+i, bc*2+j]
                
                # U_base[0, s] = coeff for A_mat[0,0]
                # U_base[1, s] = coeff for A_mat[0,1]
                # U_base[2, s] = coeff for A_mat[1,0]
                # U_base[3, s] = coeff for A_mat[1,1]
                
                for (br, bc), a_coeff in A_blocks.items():
                    for i in range(2):
                        for j in range(2):
                            # A_mat[i, j] gets contribution a_coeff * A[br*2+i, bc*2+j]
                            # U_base[i*2+j, s] is the coefficient for A_mat[i, j]
                            U[br*2+i, bc*2+j, col] += a_coeff * U_base[i*2+j, s]
                
                # Similarly for B
                for (br, bc), b_coeff in B_blocks.items():
                    for i in range(2):
                        for j in range(2):
                            # B_mat[i, j] gets contribution b_coeff * B[br*2+i, bc*2+j]
                            # V_base[i*2+j, s] is the coefficient for B_mat[i, j]
                            V[br*2+i, bc*2+j, col] += b_coeff * V_base[i*2+j, s]
                
                # For W: the result P_t is a 2x2 matrix.
                # P_t[i, j] contributes to C-blocks based on C_blocks.
                # C_blocks maps block_pos to sign.
                # P_t is placed at each block position in C_blocks with the given sign.
                
                # W_base[i*2+j, s] is the coefficient for P_t[i, j] in the s-th sub-term.
                # P_t[i, j] contributes to C[br*2+i, bc*2+j] for each (br, bc) in C_blocks with sign.
                
                for (br, bc), c_coeff in C_blocks.items():
                    for i in range(2):
                        for j in range(2):
                            # C entry at (br*2+i, bc*2+j)
                            # In our indexing, C is indexed by (k, i) where k is row of B (0..3), i is col of A (0..3)
                            # Wait, the convention is W[k*n+i, r] where k is the column index of B (0..p-1) and i is the row index of A (0..n-1)
                            # Actually, C = A * B, so C[i, k] = sum_j A[i,j] * B[j,k]
                            # The output factor W[k*n+i, r] means W is indexed by (k, i) where k is column of C (0..p-1) and i is row of C (0..n-1)
                            # So C[i, k] corresponds to W[k*n + i, :]
                            
                            # In our block structure:
                            # C_block(br, bc) has entries C[br*2+i, bc*2+j] for i,j in {0,1}
                            # So C[br*2+i, bc*2+j] corresponds to W[(bc*2+j)*n + (br*2+i), col]
                            
                            W[(bc*2+j)*n + (br*2+i), col] += c_coeff * W_base[i*2+j, s]
        
        return U, V, W

    # Build the 4x4 * 4x4 decomposition
    U_444, V_444, W_444 = build_4x4x4()
    
    # Now for 4x4 * 4x8:
    # B is 4x8. Split into B_left (4x4) and B_right (4x4).
    # C = A * B = [A * B_left | A * B_right]
    # So we do two 4x4 * 4x4 multiplications.
    
    # For the left half (columns 0-3 of B):
    # U_left = U_444 (same A)
    # V_left: B entries for columns 0-3
    # W_left: C entries for columns 0-3
    
    # For the right half (columns 4-7 of B):
    # U_right = U_444 (same A)
    # V_right: B entries for columns 4-7
    # W_right: C entries for columns 4-7
    
    # V_444 has shape (16, 49) where V[j*4+k, col] corresponds to B[j, k] for j in 0..3, k in 0..3
    # For the left half, B[j, k] for k in 0..3 is the same.
    # For the right half, B[j, k] for k in 4..7. In the 4x4 sub-problem, this is B[j, k-4].
    
    # W_444 has shape (16, 49) where W[k*4+i, col] corresponds to C[i, k] for i in 0..3, k in 0..3
    # For the left half, C[i, k] for k in 0..3 is the same.
    # For the right half, C[i, k] for k in 4..7. In the 4x4 sub-problem, this is C[i, k-4].
    
    rank_total = 98
    U_total = np.zeros((n * m, rank_total))
    V_total = np.zeros((m * p, rank_total))
    W_total = np.zeros((p * n, rank_total))
    
    # Left half
    U_total[:, 0:49] = U_444
    # V for left half: V_444[j*4+k, col] -> V_total[j*8+k, col] for k in 0..3
    for j in range(4):
        for k in range(4):
            V_total[j*8+k, 0:49] = V_444[j*4+k, :]
    # W for left half: W_444[k*4+i, col] -> W_total[k*4+i, col] for k in 0..3
    for k in range(4):
        for i in range(4):
            W_total[k*4+i, 0:49] = W_444[k*4+i, :]
    
    # Right half
    U_total[:, 49:98] = U_444
    # V for right half: V_444[j*4+k, col] -> V_total[j*8+(k+4), col] for k in 0..3
    for j in range(4):
        for k in range(4):
            V_total[j*8+(k+4), 49:98] = V_444[j*4+k, :]
    # W for right half: W_444[k*4+i, col] -> W_total[(k+4)*4+i, col] for k in 0..3
    for k in range(4):
        for i in range(4):
            W_total[(k+4)*4+i, 49:98] = W_444[k*4+i, :]
    
    # Verify the decomposition
    def verify(U, V, W):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # The tensor T[i, j, k] = 1 if i, j, k are valid indices, else 0
        # Actually, the matrix multiplication tensor is:
        # T[i, j, k] = delta (always 1 for valid i, j, k)
        # The decomposition is: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's check for all (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0  # The tensor value
                    rhs = 0.0
                    for r in range(rank):
                        rhs += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(lhs - rhs) > 1e-9:
                        return False, (i, j, k, lhs, rhs)
        return True, None
    
    is_valid, info = verify(U_total, V_total, W_total)
    
    if not is_valid:
        # Fallback to schoolbook
        result = schoolbook()
        return result
    
    # Check time
    if time.time() > deadline:
        result = schoolbook()
        return result
    
    return {"U": U_total, "V": V_total, "W": W_total}
