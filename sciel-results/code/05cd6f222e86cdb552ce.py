import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x6 matrix B into three 3x2 blocks: B1, B2, B3
    # C = A * B = A * [B1, B2, B3] = [A*B1, A*B2, A*B3]
    # Each A*Bi is a 2x2 matrix product.
    # We use Strassen's algorithm for each 2x2 product.
    # Strassen rank for 2x2x2 is 7.
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for A (2x3):
    # A is 2x3. We need to map 2x2 sub-blocks.
    # For block 0 (cols 0,1 of B):
    # B1 is 3x2. A is 2x3.
    # Wait, A is 2x3. B1 is 3x2.
    # The product is 2x2.
    # The "2x2" Strassen assumes A is 2x2.
    # Here A has 3 columns.
    # We cannot directly use Strassen on A (2x3) and B (3x2).
    # Strassen requires square matrices.
    
    # So the 21 rank idea is flawed because A is not 2x2.
    # We must use a decomposition for 2x3x2.
    # Schoolbook for 2x3x2 is 12.
    # Total rank 3 * 12 = 36.
    
    # Let's try to find a better decomposition for 2x3x2.
    # If we can't, we return schoolbook 36.
    
    # However, we can split A into 2x1 and 2x2?
    # A = [A1, A2] where A1 is 2x1, A2 is 2x2.
    # B = [B1; B2] where B1 is 1x2, B2 is 2x2.
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x1 * 1x2 -> 2x2. Rank 4?
    # A2*B2 is 2x2 * 2x2 -> 2x2. Rank 7.
    # Total 11?
    # A1*B1:
    # A1 = [[a0], [a1]], B1 = [b0, b1].
    # C = [[a0*b0, a0*b1], [a1*b0, a1*b1]].
    # This is 4 terms.
    # A2*B2:
    # A2 = [[a2, a3], [a4, a5]], B2 = [[b2, b3], [b4, b5]].
    # Strassen 7 terms.
    # Total 11 terms.
    
    # Let's implement this 11-term decomposition for 2x3x2.
    # Then total rank for 2x3x6 is 3 * 11 = 33.
    
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_offset = 0
    for block in range(3):
        # B block columns: 2*block, 2*block+1
        b_col_0 = 2 * block
        b_col_1 = 2 * block + 1
        
        # A columns: 0, 1, 2
        # A1 is col 0, A2 is cols 1,2
        # B1 is row 0, B2 is rows 1,2
        
        # Part 1: A1 * B1 (2x1 * 1x2)
        # A1 = [[A00], [A10]]
        # B1 = [B00, B01]
        # Terms:
        # T1: A00 * B00 -> C00
        # T2: A00 * B01 -> C01
        # T3: A10 * B00 -> C10
        # T4: A10 * B01 -> C11
        
        U[0][col_offset] = 1.0; V[0][col_offset] = 1.0; W[0*2+0][col_offset] = 1.0; col_offset += 1
        U[0][col_offset] = 1.0; V[1][col_offset] = 1.0; W[1*2+0][col_offset] = 1.0; col_offset += 1
        U[3][col_offset] = 1.0; V[0][col_offset] = 1.0; W[0*2+1][col_offset] = 1.0; col_offset += 1
        U[3][col_offset] = 1.0; V[1][col_offset] = 1.0; W[1*2+1][col_offset] = 1.0; col_offset += 1
        
        # Part 2: A2 * B2 (2x2 * 2x2)
        # A2 = [[A01, A02], [A11, A12]]
        # B2 = [[B10, B11], [B20, B21]]
        # Strassen:
        # M1 = (A01+A12)(B10+B21)
        # M2 = (A11+A12)B10
        # M3 = A01(B11-B21)
        # M4 = A12(B20-B11)
        # M5 = (A01+A11)B21
        # M6 = (A11-A01)(B10+B11)
        # M7 = (A02-A12)(B20+B21)
        
        # M1
        U[1][col_offset] = 1.0; U[5][col_offset] = 1.0
        V[3][col_offset] = 1.0; V[7][col_offset] = 1.0
        W[0*2+0][col_offset] = 1.0; W[1*2+1][col_offset] = 1.0
        col_offset += 1
        
        # M2
        U[4][col_offset] = 1.0; U[5][col_offset] = 1.0
        V[3][col_offset] = 1.0
        W[0*2+1][col_offset] = -1.0; W[1*2+1][col_offset] = 1.0 # C21 += M2, C22 -= M2?
        # C21 = M2+M4. C22 = M1-M2+M3+M6.
        # So W for M2: C21 += 1, C22 -= 1.
        # Indices: C21 is W[1*2+0]? No.
        # C is 2x2.
        # C00, C01, C10, C11.
        # W index: k*n + i.
        # For block 0, k=0,1.
        # C00: k=0, i=0 -> W[0]
        # C01: k=1, i=0 -> W[2]
        # C10: k=0, i=1 -> W[1]
        # C11: k=1, i=1 -> W[3]
        
        # Let's re-map W indices for the block.
        # Block 0:
        # C00: W[0*2+0] = W[0]
        # C10: W[0*2+1] = W[1]
        # C01: W[1*2+0] = W[2]
        # C11: W[1*2+1] = W[3]
        
        # My previous assignments:
        # T1: A00*B00 -> C00. W[0]. Correct.
        # T2: A00*B01 -> C01. W[2]. Correct.
        # T3: A10*B00 -> C10. W[1]. Correct.
        # T4: A10*B01 -> C11. W[3]. Correct.
        
        # Now Strassen for A2*B2:
        # C00 += M1 + M4 - M5 + M7
        # C01 += M3 + M5
        # C10 += M2 + M4
        # C11 += M1 - M2 + M3 + M6
        
        # M1
        W[0][col_offset-1] = 1.0 # C00
        W[3][col_offset-1] = 1.0 # C11
        
        # M2
        W[1][col_offset] = 1.0 # C10
        W[3][col_offset] = -1.0 # C11
        
        # M3
        W[2][col_offset] = 1.0 # C01
        W[3][col_offset] = 1.0 # C11
        
        # M4
        W[0][col_offset] = 1.0 # C00
        W[1][col_offset] = 1.0 # C10
        
        # M5
        W[0][col_offset] = -1.0 # C00
        W[2][col_offset] = 1.0 # C01
        
        # M6
        W[3][col_offset] = 1.0 # C11
        
        # M7
        W[0][col_offset] = 1.0 # C00
        
        # Fill U, V for M2..M7
        # M2
        U[4][col_offset] = 1.0; U[5][col_offset] = 1.0
        V[3][col_offset] = 1.0
        col_offset += 1
        
        # M3
        U[1][col_offset] = 1.0
        V[4][col_offset] = 1.0; V[7][col_offset] = -1.0
        col_offset += 1
        
        # M4
        U[5][col_offset] = 1.0
        V[6][col_offset] = 1.0; V[4][col_offset] = -1.0
        col_offset += 1
        
        # M5
        U[1][col_offset] = 1.0; U[4][col_offset] = 1.0
        V[7][col_offset] = 1.0
        col_offset += 1
        
        # M6
        U[4][col_offset] = 1.0; U[1][col_offset] = -1.0
        V[3][col_offset] = 1.0; V[4][col_offset] = 1.0
        col_offset += 1
        
        # M7
        U[2][col_offset] = 1.0; U[5][col_offset] = -1.0
        V[6][col_offset] = 1.0; V[7][col_offset] = 1.0
        col_offset += 1
        
        # Note: V indices for B2 (rows 1,2 of B block)
        # B block is 3x2.
        # B1 is row 0. B2 is rows 1,2.
        # B indices in global V:
        # Block 0: B rows 0,1,2. Cols 0,1.
        # V index = j*p + k.
        # j=0, k=0,1 -> V[0], V[1]
        # j=1, k=0,1 -> V[3], V[4]
        # j=2, k=0,1 -> V[6], V[7]
        
        # My V assignments:
        # B1 (row 0): V[0], V[1]. Correct.
        # B2 (rows 1,2):
        # B10 (j=1,k=0) -> V[3]
        # B11 (j=1,k=1) -> V[4]
        # B20 (j=2,k=0) -> V[6]
        # B21 (j=2,k=1) -> V[7]
        
        # M1: V[3], V[7]. Correct.
        # M2: V[3]. Correct.
        # M3: V[4], V[7]. Correct.
        # M4: V[6], V[4]. Correct.
        # M5: V[7]. Correct.
        # M6: V[3], V[4]. Correct.
        # M7: V[6], V[7]. Correct.
        
        # A indices:
        # A1 (col 0): U[0], U[3].
        # A2 (cols 1,2):
        # A01 (i=0,j=1) -> U[1]
        # A02 (i=0,j=2) -> U[2]
        # A11 (i=1,j=1) -> U[4]
        # A12 (i=1,j=2) -> U[5]
        
        # M1: U[1], U[5]. Correct.
        # M2: U[4], U[5]. Correct.
        # M3: U[1]. Correct.
        # M4: U[5]. Correct.
        # M5: U[1], U[4]. Correct.
        # M6: U[4], U[1]. Correct.
        # M7: U[2], U[5]. Correct.
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
