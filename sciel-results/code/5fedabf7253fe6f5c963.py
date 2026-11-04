import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # B is 3x9, split into 3 blocks of 3x3.
    # For each block, we compute A (2x3) * B_block (3x3).
    # A is split into A_2x2 (cols 0,1) and a_2x1 (col 2).
    # B_block is split into B_2x2 (rows 0,1) and b_1x2 (row 2).
    # Product = A_2x2 * B_2x2 + a_2x1 * b_1x2.
    # Rank of A_2x2 * B_2x2 is 8 (schoolbook).
    # Rank of a_2x1 * b_1x2 is 4 (schoolbook).
    # Total rank per block = 12.
    # Total rank = 3 * 12 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 6 x 36
    V = [[0] * rank for _ in range(m * p)] # 27 x 36
    W = [[0] * rank for _ in range(p * n)] # 18 x 36
    
    c_idx = 0
    
    # Indices in U for A:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    for block in range(3):
        # B block indices in V:
        # B[0, 3*block] = 0*9 + 3*block
        # B[0, 3*block+1] = 0*9 + 3*block+1
        # B[0, 3*block+2] = 0*9 + 3*block+2
        # B[1, 3*block] = 1*9 + 3*block
        # B[1, 3*block+1] = 1*9 + 3*block+1
        # B[1, 3*block+2] = 1*9 + 3*block+2
        # B[2, 3*block] = 2*9 + 3*block
        # B[2, 3*block+1] = 2*9 + 3*block+1
        # B[2, 3*block+2] = 2*9 + 3*block+2
        
        # W indices for C block (cols 3*block, 3*block+1, 3*block+2):
        # C[0, 3*block] = (3*block)*2 + 0
        # C[0, 3*block+1] = (3*block+1)*2 + 0
        # C[0, 3*block+2] = (3*block+2)*2 + 0
        # C[1, 3*block] = (3*block)*2 + 1
        # C[1, 3*block+1] = (3*block+1)*2 + 1
        # C[1, 3*block+2] = (3*block+2)*2 + 1
        
        # Part 1: A_2x2 * B_2x2 (8 terms)
        # A_2x2: U[0], U[1], U[3], U[4]
        # B_2x2: V[3b], V[3b+1], V[9+3b], V[9+3b+1]
        
        # C_00 = A00 B00 + A01 B10
        # C_01 = A00 B01 + A01 B11
        # C_10 = A10 B00 + A11 B10
        # C_11 = A10 B01 + A11 B11
        
        # Term 1: A00 B00
        U[0][c_idx] = 1
        V[3*block][c_idx] = 1
        W[(3*block)*2 + 0][c_idx] = 1
        W[(3*block)*2 + 1][c_idx] = 0 # C10 doesn't get A00B00
        # Wait, C is 2x3.
        # C[0, col] gets A00 B00.
        # C[1, col] gets A10 B00.
        # So W index for C[0, 3b] is (3b)*2 + 0.
        # W index for C[1, 3b] is (3b)*2 + 1.
        
        # Let's refine the W mapping.
        # C[i, k] is at W[k*n + i] = W[k*2 + i].
        
        # Term 1: A00 * B00 (contributes to C[0, 3b] and C[1, 3b]? No.
        # C[0, 3b] += A[0,0] B[0, 3b]
        # C[1, 3b] += A[1,0] B[0, 3b]
        # So A[0,0] only affects C[0, ...]. A[1,0] affects C[1, ...].
        
        # Term 1: U[0] (A00) * V[3b] (B00) -> W[(3b)*2+0] (C0, 3b)
        U[0][c_idx] = 1
        V[3*block][c_idx] = 1
        W[(3*block)*2 + 0][c_idx] = 1
        c_idx += 1
        
        # Term 2: U[3] (A10) * V[3b] (B00) -> W[(3b)*2+1] (C1, 3b)
        U[3][c_idx] = 1
        V[3*block][c_idx] = 1
        W[(3*block)*2 + 1][c_idx] = 1
        c_idx += 1
        
        # Term 3: U[0] (A00) * V[3b+1] (B01) -> W[(3b+1)*2+0] (C0, 3b+1)
        U[0][c_idx] = 1
        V[3*block+1][c_idx] = 1
        W[(3*block+1)*2 + 0][c_idx] = 1
        c_idx += 1
        
        # Term 4: U[3] (A10) * V[3b+1] (B01) -> W[(3b+1)*2+1] (C1, 3b+1)
        U[3][c_idx] = 1
        V[3*block+1][c_idx] = 1
        W[(3*block+1)*2 + 1][c_idx] = 1
        c_idx += 1
        
        # Term 5: U[1] (A01) * V[9+3b] (B10) -> W[(3b)*2+0] (C0, 3b)
        U[1][c_idx] = 1
        V[9+3*block][c_idx] = 1
        W[(3*block)*2 + 0][c_idx] = 1
        c_idx += 1
        
        # Term 6: U[4] (A11) * V[9+3b] (B10) -> W[(3b)*2+1] (C1, 3b)
        U[4][c_idx] = 1
        V[9+3*block][c_idx] = 1
        W[(3*block)*2 + 1][c_idx] = 1
        c_idx += 1
        
        # Term 7: U[1] (A01) * V[9+3b+1] (B11) -> W[(3b+1)*2+0] (C0, 3b+1)
        U[1][c_idx] = 1
        V[9+3*block+1][c_idx] = 1
        W[(3*block+1)*2 + 0][c_idx] = 1
        c_idx += 1
        
        # Term 8: U[4] (A11) * V[9+3b+1] (B11) -> W[(3b+1)*2+1] (C1, 3b+1)
        U[4][c_idx] = 1
        V[9+3*block+1][c_idx] = 1
        W[(3*block+1)*2 + 1][c_idx] = 1
        c_idx += 1
        
        # Part 2: a_2x1 * b_1x2 (4 terms)
        # a = [A02, A12] = U[2], U[5]
        # b = [B20, B21] = V[18+3b], V[18+3b+1]
        # Result:
        # C[0, 3b] += A02 B20
        # C[0, 3b+1] += A02 B21
        # C[1, 3b] += A12 B20
        # C[1, 3b+1] += A12 B21
        
        # Term 9: U[2] (A02) * V[18+3b] (B20) -> W[(3b)*2+0] (C0, 3b)
        U[2][c_idx] = 1
        V[18+3*block][c_idx] = 1
        W[(3*block)*2 + 0][c_idx] = 1
        c_idx += 1
        
        # Term 10: U[5] (A12) * V[18+3b] (B20) -> W[(3b)*2+1] (C1, 3b)
        U[5][c_idx] = 1
        V[18+3*block][c_idx] = 1
        W[(3*block)*2 + 1][c_idx] = 1
        c_idx += 1
        
        # Term 11: U[2] (A02) * V[18+3b+1] (B21) -> W[(3b+1)*2+0] (C0, 3b+1)
        U[2][c_idx] = 1
        V[18+3*block+1][c_idx] = 1
        W[(3*block+1)*2 + 0][c_idx] = 1
        c_idx += 1
        
        # Term 12: U[5] (A12) * V[18+3b+1] (B21) -> W[(3b+1)*2+1] (C1, 3b+1)
        U[5][c_idx] = 1
        V[18+3*block+1][c_idx] = 1
        W[(3*block+1)*2 + 1][c_idx] = 1
        c_idx += 1
        
        # What about the 3rd column of the block (3b+2)?
        # My split was B_2x2 (cols 0,1 of block) and b_1x2 (row 2, cols 0,1 of block).
        # This covers C[0, 3b], C[0, 3b+1], C[1, 3b], C[1, 3b+1].
        # It does NOT cover C[0, 3b+2] and C[1, 3b+2].
        
        # I need to handle the 3rd column of B_block.
        # Let B_col2 = [B02, B12, B22].
        # C[0, 3b+2] = A00 B02 + A01 B12 + A02 B22
        # C[1, 3b+2] = A10 B02 + A11 B12 + A12 B22
        # This is a 2x3x1 multiplication. Rank 6.
        
        # So total rank per block is 12 + 6 = 18.
        # Total rank 54. No improvement.
        
        # I need to optimize the 3rd column too.
        # Can I combine the 3rd column with the others?
        
        # Let's restart the block decomposition strategy.
        # Split B (3x9) into three 3x3 blocks.
        # For each 3x3 block, use a rank 11 decomposition?
        # If I can find a rank 11 decomposition for 2x3x3, total rank is 33.
        
        # I will implement a known Rank 11 decomposition for 2x3x3.
        # Reference: Bini et al.
        
        # For now, I will return the schoolbook 54 to be safe, as implementing Rank 11 correctly is risky.
        # However, I can try Rank 45 by using Rank 15 for each block?
        # Schoolbook 2x3x3 is 18.
        
        # Let's just return the schoolbook 54.
        pass
        
    # Fallback to Schoolbook 54
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
# EVOLVE-BLOCK-END
