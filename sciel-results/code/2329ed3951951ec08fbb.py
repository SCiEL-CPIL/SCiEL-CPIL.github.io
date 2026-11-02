"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Strassen 2x2x2 rank-7 decomposition
    # C = A*B where A,B,C are 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication, rank 7.
        U: shape (4, 7) - A factors: rows indexed by (i,j) in row-major
        V: shape (4, 7) - B factors: rows indexed by (i,j) in row-major  
        W: shape (4, 7) - C factors: rows indexed by (i,j) in row-major
        """
        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B00=0, B01=1, B10=2, B11=3
        # C indices: C00=0, C01=1, C10=2, C11=3
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    # Build 4x4x4 decomposition from Strassen
    # Split A into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # Split B into 2x2 blocks of 2x2: B = [[B00, B01], [B10, B11]]
    # C = A*B, split into 2x2 blocks of 2x2
    
    def compose_strassen_4x4():
        """Return U, V, W for 4x4x4 matrix multiplication, rank 49.
        U: shape (16, 49) - A factors
        V: shape (16, 49) - B factors  
        W: shape (16, 49) - C factors
        """
        S_U, S_V, S_W = strassen_2x2()
        
        # 7 Strassen products at block level
        # Each block product is 2x2x2, decomposed with rank 7
        # Total: 7 * 7 = 49
        
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Block level Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # A block indices: A00 = rows 0-1, cols 0-1; A01 = rows 0-1, cols 2-3
        #                 A10 = rows 2-3, cols 0-1; A11 = rows 2-3, cols 2-3
        # In 4x4 row-major: A00 = [0,1,4,5], A01 = [2,3,6,7], A10 = [8,9,12,13], A11 = [10,11,14,15]
        
        # B block indices: B00 = [0,1,4,5], B01 = [2,3,6,7], B10 = [8,9,12,13], B11 = [10,11,14,15]
        
        # C block indices: C00 = [0,1,4,5], C01 = [2,3,6,7], C10 = [8,9,12,13], C11 = [10,11,14,15]
        
        # For each Strassen block product k (0-6), we have a 2x2x2 multiplication
        # with its own rank-7 decomposition
        
        # Block A indices for each Strassen term
        A_blocks = [
            {0: 1, 1: 1},   # M1: A00 + A11
            {0: 0, 1: 0, 2: 1, 3: 1},  # M2: A10 + A11
            {0: 1},          # M3: A00
            {0: 0, 1: 0, 2: 1, 3: 1},  # M4: A11
            {0: 1, 1: 1},   # M5: A00 + A01
            {0: 0, 1: -1, 2: 1, 3: -1}, # M6: A10 - A00
            {0: 0, 1: 0, 2: -1, 3: -1}, # M7: A01 - A11
        ]
        
        # Block B indices for each Strassen term
        B_blocks = [
            {0: 1, 1: 1},   # M1: B00 + B11
            {0: 1},          # M2: B00
            {0: 0, 1: 0, 2: 1, 3: -1}, # M3: B10 - B11
            {0: 0, 1: -1, 2: 1, 3: 1}, # M4: B01 - B00
            {0: 1},          # M5: B11
            {0: 1, 1: 1},   # M6: B00 + B01
            {0: 0, 1: 0, 2: 1, 3: 1},  # M7: B10 + B11
        ]
        
        # C block contributions for each Strassen term
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        C_contrib = [
            {0: 1, 1: 0, 2: 0, 3: 1},   # M1
            {0: 0, 1: 0, 2: 1, 3: -1},  # M2
            {0: 0, 1: 1, 2: 0, 3: 1},   # M3
            {0: 1, 1: 0, 2: 1, 3: 0},   # M4
            {0: -1, 1: 1, 2: 0, 3: 1},  # M5
            {0: 0, 1: 0, 2: 0, 3: 1},   # M6
            {0: 1, 1: 0, 2: 0, 3: 0},   # M7
        ]
        
        # Map block index to actual row/col indices in 4x4
        # Block 0 = A00: rows 0,1 cols 0,1 -> indices 0,1,4,5
        # Block 1 = A01: rows 0,1 cols 2,3 -> indices 2,3,6,7
        # Block 2 = A10: rows 2,3 cols 0,1 -> indices 8,9,12,13
        # Block 3 = A11: rows 2,3 cols 2,3 -> indices 10,11,14,15
        block_to_indices = {
            0: [0, 1, 4, 5],
            1: [2, 3, 6, 7],
            2: [8, 9, 12, 13],
            3: [10, 11, 14, 15]
        }
        
        for k in range(7):
            # For Strassen block product k, decompose the 2x2x2 multiplication
            base_col = k * 7
            for t in range(7):  # rank-7 sub-decomposition
                col = base_col + t
                
                # U contribution: for each A index in sub-decomposition
                for a_idx in range(4):
                    u_val = S_U[a_idx][t]
                    if u_val != 0:
                        # Which block of A does a_idx correspond to?
                        # In 2x2: a_idx 0=A00, 1=A01, 2=A10, 3=A11
                        # Find all blocks in A_blocks[k] that contribute to this a_idx
                        for block_id, coeff in A_blocks[k].items():
                            # In the 2x2 sub-problem, a_idx maps to a 2x2 matrix position
                            # a_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                            # We need to map to actual 4x4 indices
                            pass
                
                # Simpler approach: directly build by expanding
                # For each of the 4 entries in the 2x2 A matrix (position a_idx)
                # we need to know which 2x2 block it comes from and what coefficient
                pass
        
        # Let me redo this more carefully
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for k in range(7):
            base_col = k * 7
            for t in range(7):
                col = base_col + t
                
                # U: S_U[a_idx][t] for a_idx in 0..3
                for a_idx in range(4):
                    u_val = S_U[a_idx][t]
                    if u_val == 0:
                        continue
                    # a_idx corresponds to position in 2x2: 
                    # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    # In the block-level Strassen, which A block contributes?
                    # A_blocks[k] maps block_id -> coefficient
                    # But we need: for the 2x2 A matrix in this Strassen product,
                    # entry (r,c) = sum over blocks that map to position (r,c)
                    
                    r = a_idx // 2
                    c = a_idx % 2
                    
                    for block_id, coeff in A_blocks[k].items():
                        # block_id 0->A00(pos 0,0), 1->A01(pos 0,1), 2->A10(pos 1,0), 3->A11(pos 1,1)
                        br = block_id // 2
                        bc = block_id % 2
                        if br == r and bc == c:
                            # This block contributes with coefficient coeff
                            for actual_idx in block_to_indices[block_id]:
                                U[actual_idx][col] += u_val * coeff
                            break
                
                # V: S_V[b_idx][t] for b_idx in 0..3
                for b_idx in range(4):
                    v_val = S_V[b_idx][t]
                    if v_val == 0:
                        continue
                    r = b_idx // 2
                    c = b_idx % 2
                    
                    for block_id, coeff in B_blocks[k].items():
                        br = block_id // 2
                        bc = block_id % 2
                        if br == r and bc == c:
                            for actual_idx in block_to_indices[block_id]:
                                V[actual_idx][col] += v_val * coeff
                            break
                
                # W: S_W[c_idx][t] for c_idx in 0..3
                for c_idx in range(4):
                    w_val = S_W[c_idx][t]
                    if w_val == 0:
                        continue
                    r = c_idx // 2
                    c = c_idx % 2
                    
                    for block_id, coeff in C_contrib[k].items():
                        br = block_id // 2
                        bc = block_id % 2
                        if br == r and bc == c:
                            for actual_idx in block_to_indices[block_id]:
                                W[actual_idx][col] += w_val * coeff
                            break
        
        return U, V, W
    
    # Now build 4x4x8 from two 4x4x4 decompositions
    # A is 4x4, B is 4x8, C is 4x8
    # Split B into B_left (4x4) and B_right (4x4)
    # C_left = A * B_left, C_right = A * B_right
    
    def construct_4x4x8():
        U4, V4, W4 = compose_strassen_4x4()
        
        rank = 98
        U = [[0]*rank for _ in range(16)]  # 4*4
        V = [[0]*rank for _ in range(32)]  # 4*8
        W = [[0]*rank for _ in range(32)]  # 8*4, but trace convention: W[k*n+i, r] where k is output row, i is input col? 
        # Wait, let me re-read: W[k*n+i, r] where n=4 (rows of C), so W has p*n = 8*4 = 32 rows
        # k ranges 0..7 (column of B / row of C?), i ranges 0..3
        # Actually for 4x4 * 4x8: A is 4x4, B is 4x8, C is 4x8
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # W index: k*n + i where k is the B-row/C-col index (0..7) and i is the A-row/C-row index (0..3)
        # So W[k*4 + i, r]
        
        # Left half: columns 0-3 of B, columns 0-3 of C
        # Right half: columns 4-7 of B, columns 4-7 of C
        
        for r in range(49):
            # Left part
            for i in range(16):
                U[i][r] = U4[i][r]
            for j in range(16):
                # B indices: j=0..3 -> B rows 0..3 for left half (B[k, 0..3])
                # In 4x4x4, V[j] corresponds to B[j] where j is row-major in 4x4
                # For left half, B[k, col] where col=0..3, so B indices are k*4 + col
                # But in 4x4x4 V, index j corresponds to B[j] in the 4x4 B
                # For left: B is 4x4, so V4[j] maps to B_left[j]
                # B_left has columns 0-3 of the full 4x8 B
                # In 4x8, B[k, c] where c=0..7. Left: c=0..3
                # V4 index j (0..15) in 4x4: row = j//4, col = j%4
                # Full B: row = j//4, col = j%4 (for left)
                b_row = j // 4
                b_col = j % 4
                full_b_idx = b_row * 8 + b_col
                V[full_b_idx][r] = V4[j][r]
            for c in range(16):
                # W4 index c (0..15) in 4x4x4: C[row, col] where row=c//4, col=c%4
                # For left half, C[row, col] where col=0..3
                # W index: k*n + i where k = C col (0..7 for full), i = C row (0..3)
                # For left: k = col (0..3)
                c_row = c // 4
                c_col = c % 4
                w_idx = c_col * 4 + c_row
                W[w_idx][r] = W4[c][r]
        
            # Right part
            for i in range(16):
                U[i][49 + r] = U4[i][r]
            for j in range(16):
                b_row = j // 4
                b_col = j % 4
                full_b_idx = b_row * 8 + (b_col + 4)
                V[full_b_idx][49 + r] = V4[j][r]
            for c in range(16):
                c_row = c // 4
                c_col = c % 4
                w_idx = (c_col + 4) * 4 + c_row
                W[w_idx][49 + r] = W4[c][r]
        
        return U, V, W
    
    if time.time() > deadline:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = construct_4x4x8()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

