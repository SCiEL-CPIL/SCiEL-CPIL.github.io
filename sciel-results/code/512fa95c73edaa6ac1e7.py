"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Helper to append a decomposition to U, V, W
    # U, V, W are lists of lists.
    # offset is the current column index.
    def add_decomp(U, V, W, u_rows, v_rows, w_rows, u_indices, v_indices, w_indices, u_vals, v_vals, w_vals, offset):
        rank = len(u_vals)
        for c in range(rank):
            for r in range(u_rows):
                if u_indices[r] is not None:
                    U[u_indices[r]][offset + c] += u_vals[r][c]
            for r in range(v_rows):
                if v_indices[r] is not None:
                    V[v_indices[r]][offset + c] += v_vals[r][c]
            for r in range(w_rows):
                if w_indices[r] is not None:
                    W[w_indices[r]][offset + c] += w_vals[r][c]
        return offset + rank

    # Strassen 2x2 decomposition (Rank 7)
    # Inputs: U (2x2), V (2x2), W (2x2)
    # Indices in 2x2: 0,1,2,3
    # U[i,j], V[j,k], W[k,i]
    # Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to map these to the tensor decomposition format.
    # The tensor decomposition is a sum of rank-1 tensors.
    # Strassen's algorithm is a sum of 7 matrix products of linear combinations.
    # This is NOT a direct sum of rank-1 tensors in the standard sense unless we expand the linear combinations.
    # Wait. The rank of the tensor is the minimum number of rank-1 tensors.
    # Strassen's algorithm shows that the tensor can be computed with 7 scalar multiplications.
    # This implies the rank is at most 7.
    # However, the "rank-1 tensors" in Strassen's algorithm are of the form:
    # $u_c \otimes v_c \otimes w_c$ where $u_c, v_c$ are vectors of coefficients for the linear combinations.
    # So we can just extract the coefficients.
    
    # For 2x2:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 7 multiplications are:
    # 1. (A11 + A22) * (B11 + B22)
    # 2. (A21 + A22) * B11
    # 3. A11 * (B12 - B22)
    # 4. A22 * (B21 - B11)
    # 5. (A11 + A12) * B22
    # 6. (A21 - A11) * (B11 + B12)
    # 7. (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to construct U, V, W for the 2x2 case.
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    # Indices: 0=A11, 1=A12, 2=A21, 3=A22.
    
    U2 = zeros(4, 7)
    V2 = zeros(4, 7)
    W2 = zeros(4, 7)
    
    # M1: (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C11 gets +1, C22 gets +1?
    # C11 = M1 ... C22 = M1 ...
    # W[k*n+i, c]
    # For 2x2, n=2.
    # C11 is index 0. C12 is 1. C21 is 2. C22 is 3.
    # W[0, 0] += 1 (for C11)
    # W[3, 0] += 1 (for C22)
    W2[0][0] = 1
    W2[3][0] = 1
    
    # M2: (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1 # C21
    W2[3][1] = -1 # C22
    
    # M3: A11(B12-B22)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1 # C12
    W2[3][2] = 1 # C22
    
    # M4: A22(B21-B11)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1 # C11
    W2[2][3] = 1 # C21
    
    # M5: (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1 # C11
    W2[1][4] = 1 # C12
    
    # M6: (A21-A11)(B11+B12)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C22
    
    # M7: (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C11
    W2[3][6] = 1 # C22
    
    # Now build 4x4 decomposition.
    # Rank 56.
    U4 = zeros(16, 56)
    V4 = zeros(16, 56)
    W4 = zeros(16, 56)
    
    offset = 0
    
    # Diagonal blocks: A11 (0:4), A22 (12:16)
    # B11 (0:4), B22 (12:16)
    # C11 (0:4), C22 (12:16)
    
    # Map 2x2 indices to 4x4 indices.
    # 2x2 index 0 -> 4x2 0
    # 2x2 index 1 -> 4x2 1
    # 2x2 index 2 -> 4x2 4
    # 2x2 index 3 -> 4x2 5
    
    def map_2x2_to_4x2(base, idx):
        # base is 0 for top-left, 12 for bottom-right?
        # 4x4 matrix is 4x4.
        # Top-left 2x2: rows 0,1; cols 0,1.
        # Bottom-right 2x2: rows 2,3; cols 2,3.
        # Index in 2x2: 0,1,2,3.
        # 0 -> row 0, col 0
        # 1 -> row 0, col 1
        # 2 -> row 1, col 0
        # 3 -> row 1, col 1
        # If base is 0:
        # 0->0, 1->1, 2->4, 3->5
        # If base is 12 (row 2, col 2):
        # 0->12, 1->13, 2->16? No, 4x4 has 16 elements.
        # Row 2, Col 2 is index 2*4+2 = 10.
        # Row 2, Col 3 is 11.
        # Row 3, Col 2 is 14.
        # Row 3, Col 3 is 15.
        # So 0->10, 1->11, 2->14, 3->15.
        
        if base == 0:
            return [0, 1, 4, 5][idx]
        else: # base 10
            return [10, 11, 14, 15][idx]
        
    # Add Strassen for top-left
    for c in range(7):
        for r in range(4):
            u_idx = map_2x2_to_4x2(0, r)
            v_idx = map_2x2_to_4x2(0, r)
            w_idx = map_2x2_to_4x2(0, r)
            U4[u_idx][offset + c] = U2[r][c]
            V4[v_idx][offset + c] = V2[r][c]
            W4[w_idx][offset + c] = W2[r][c]
    offset += 7
    
    # Add Strassen for bottom-right
    for c in range(7):
        for r in range(4):
            u_idx = map_2x2_to_4x2(10, r)
            v_idx = map_2x2_to_4x2(10, r)
            w_idx = map_2x2_to_4x2(10, r)
            U4[u_idx][offset + c] = U2[r][c]
            V4[v_idx][offset + c] = V2[r][c]
            W4[w_idx][offset + c] = W2[r][c]
    offset += 7
    
    # Off-diagonal blocks.
    # A12 (rows 0,1; cols 2,3). Indices: 2,3,6,7.
    # A21 (rows 2,3; cols 0,1). Indices: 8,9,12,13.
    # B12 (rows 0,1; cols 2,3). Indices: 2,3,6,7.
    # B21 (rows 2,3; cols 0,1). Indices: 8,9,12,13.
    # C12 (rows 0,1; cols 2,3). Indices: 2,3,6,7.
    # C21 (rows 2,3; cols 0,1). Indices: 8,9,12,13.
    
    # We need to perform 6 Strassen decompositions.
    # 1. A12 * B21 -> C12_part
    # 2. A12 * B22 -> C12_part
    # 3. A11 * B12 -> C12_part
    # 4. A21 * B11 -> C21_part
    # 5. A21 * B12 -> C21_part
    # 6. A22 * B21 -> C21_part
    
    # Wait, the Strassen decomposition computes C = A * B.
    # So we apply it to the blocks.
    
    def add_strassen_block(u_base, v_base, w_base, offset):
        # u_base, v_base, w_base are starting indices in 4x4 matrices.
        # We need to map the 2x2 local indices to the 4x4 global indices.
        for c in range(7):
            for r in range(4):
                # Local index r in 2x2 corresponds to:
                # 0: row 0, col 0
                # 1: row 0, col 1
                # 2: row 1, col 0
                # 3: row 1, col 1
                
                # Global index calculation
                # u_idx = u_base + (r % 2) * 4 + (r // 2) ? No.
                # 2x2 index r:
                # 0 -> (0,0)
                # 1 -> (0,1)
                # 2 -> (1,0)
                # 3 -> (1,1)
                
                row = r // 2
                col = r % 2
                
                # But the block is 2x2.
                # If u_base is the index of the top-left of the block.
                # Then global index = u_base + row * 4 + col.
                
                u_idx = u_base + row * 4 + col
                v_idx = v_base + row * 4 + col
                w_idx = w_base + row * 4 + col
                
                U4[u_idx][offset + c] += U2[r][c]
                V4[v_idx][offset + c] += V2[r][c]
                W4[w_idx][offset + c] += W2[r][c]
                
        return offset + 7

    # 1. A12 * B21 -> C12
    # A12 base: 2. B21 base: 8. C12 base: 2.
    offset = add_strassen_block(2, 8, 2, offset)
    
    # 2. A12 * B22 -> C12
    # A12 base: 2. B22 base: 10. C12 base: 2.
    offset = add_strassen_block(2, 10, 2, offset)
    
    # 3. A11 * B12 -> C12
    # A11 base: 0. B12 base: 2. C12 base: 2.
    offset = add_strassen_block(0, 2, 2, offset)
    
    # 4. A21 * B11 -> C21
    # A21 base: 8. B11 base: 0. C21 base: 8.
    offset = add_strassen_block(8, 0, 8, offset)
    
    # 5. A21 * B12 -> C21
    # A21 base: 8. B12 base: 2. C21 base: 8.
    offset = add_strassen_block(8, 2, 8, offset)
    
    # 6. A22 * B21 -> C21
    # A22 base: 10. B21 base: 8. C21 base: 8.
    offset = add_strassen_block(10, 8, 8, offset)
    
    # Total rank for 4x4 is 56.
    
    # Now build 5x5.
    # Rank 109.
    U5 = zeros(25, 109)
    V5 = zeros(25, 109)
    W5 = zeros(25, 109)
    
    offset5 = 0
    
    # 1. A44 * B44 -> C44
    # A44 is top-left 4x4. Base 0.
    # B44 is top-left 4x4. Base 0.
    # C44 is top-left 4x4. Base 0.
    for c in range(56):
        for r in range(16):
            U5[r][offset5 + c] = U4[r][c]
            V5[r][offset5 + c] = V4[r][c]
            W5[r][offset5 + c] = W4[r][c]
    offset5 += 56
    
    # 2. Cross terms (Schoolbook)
    # a4 is row 4 (indices 16,17,18,19)
    # b4 is col 4 (indices 16,20,24,28? No, 5x5 matrix is 25 elements.
    # Index i*5+j.
    # Row 4: 20, 21, 22, 23, 24.
    # a4 is A[4, 0:4] -> indices 20, 21, 22, 23.
    # b4 is B[0:4, 4] -> indices 4, 9, 14, 19.
    
    # C41 = a4 * b1^T
    # a4 (1x4) * b1 (4x1) -> 4x4 matrix.
    # C[4, 0:4] = a4 * b1^T.
    # C indices: 20, 21, 22, 23.
    # a4 indices: 20, 21, 22, 23.
    # b1 indices: 0, 1, 2, 3.
    
    # 16 multiplications.
    for i in range(4):
        for j in range(4):
            u_idx = 20 + i # a4[i]
            v_idx = j # b1[j]
            w_idx = 20 + j # C[4, j]
            U5[u_idx][offset5] = 1
            V5[v_idx][offset5] = 1
            W5[w_idx][offset5] = 1
            offset5 += 1
            
    # C44_part1 = A44 * b4
    # A44 (4x4) * b4 (4x1) -> 4x1 vector.
    # C[0:4, 4] = A44 * b4.
    # C indices: 4, 9, 14, 19.
    # A44 indices: 0..15.
    # b4 indices: 4, 9, 14, 19.
    
    # 16 multiplications.
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j # A44[i, j]
            v_idx = 4 + j * 5 # b4[j] -> B[j, 4] -> index j*5+4
            w_idx = 4 + i * 5 # C[i, 4] -> index i*5+4
            U5[u_idx][offset5] = 1
            V5[v_idx][offset5] = 1
            W5[w_idx][offset5] = 1
            offset5 += 1

    # C14 = a1^T * B44
    # a1 (1x4) * B44 (4x4) -> 1x4 vector.
    # C[0, 0:4] = a1^T * B44.
    # C indices: 0, 1, 2, 3.
    # a1 indices: 0, 1, 2, 3.
    # B44 indices: 0..15.
    
    # 16 multiplications.
    for i in range(4):
        for j in range(4):
            u_idx = i # a1[i]
            v_idx = i * 4 + j # B44[i, j]
            w_idx = j # C[0, j]
            U5[u_idx][offset5] = 1
            V5[v_idx][offset5] = 1
            W5[w_idx][offset5] = 1
            offset5 += 1

    # C11_part1 = a1^T * b1
    # a1 (1x4) * b1 (4x1) -> scalar.
    # C[0, 0] += a1^T * b1.
    # C index: 0.
    # a1 indices: 0, 1, 2, 3.
    # b1 indices: 0, 1, 2, 3.
    
    # 4 multiplications.
    for i in range(4):
        u_idx = i # a1[i]
        v_idx = i # b1[i]
        w_idx = 0 # C[0, 0]
        U5[u_idx][offset5] = 1
        V5[v_idx][offset5] = 1
        W5[w_idx][offset5] = 1
        offset5 += 1

    # C11_part2 = a11 * b11
    # a11 (scalar) * b11 (scalar) -> scalar.
    # C[0, 0] += a11 * b11.
    # a11 index: 0.
    # b11 index: 0.
    # C index: 0.
    
    U5[0][offset5] = 1
    V5[0][offset5] = 1
    W5[0][offset5] = 1
    offset5 += 1

    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END
