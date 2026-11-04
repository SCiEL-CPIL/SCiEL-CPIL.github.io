import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose 6x6 into 3x3 blocks of 2x2 matrices.
    # Block size: 2
    # Number of blocks: 3
    
    # Strassen's algorithm for 2x2 matrices (Rank 7)
    # U, V, W for 2x2 multiplication
    # Indices for 2x2: 
    # A: [[0,1],[2,3]] -> row 0: 0,1; row 1: 2,3
    # B: [[0,1],[2,3]]
    # C: [[0,1],[2,3]]
    
    # Strassen 7 terms:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We will build the global U, V, W matrices.
    # Global size: 6x6.
    # Block index (bi, bj) where bi, bj in 0..2.
    # Local index (li, lj) where li, lj in 0..1.
    # Global row index for A: bi*2 + li, col: bj*2 + lj.
    # Global row index for B: bi*2 + li, col: bj*2 + lj.
    # Global row index for C: bi*2 + li, col: bj*2 + lj.
    
    # The 3x3 block multiplication is schoolbook (Rank 27).
    # For each block (I, J) in C, C[I,J] = sum_{K=0}^{2} A[I,K] * B[K,J].
    # Each "scalar" multiplication A[I,K] * B[K,J] is a 2x2 matrix multiplication.
    # We replace each scalar multiplication with the 7-term Strassen decomposition.
    # Total terms = 27 * 7 = 189.
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    col = 0
    # Iterate over block indices I, K, J
    for I in range(3):
        for K in range(3):
            for J in range(3):
                # This corresponds to one scalar multiplication in the 3x3 block product:
                # Term = A_block[I,K] * B_block[K,J]
                # We use Strassen for this 2x2 multiplication.
                
                # Strassen coefficients for U (A part), V (B part), W (C part)
                # A indices in 2x2:
                # a=0, b=1, c=2, d=3 (row-major)
                # B indices in 2x2:
                # e=0, f=1, g=2, h=3
                # C indices in 2x2:
                # c00=0, c01=1, c10=2, c11=3
                
                # Term 1: (a+d)(e+h)
                # U: a(0), d(3) -> 1
                # V: e(0), h(3) -> 1
                # W: c00(0), c11(3) -> 1, 1? No, C = M1 + M4 - M5 + M7 for c00, etc.
                # We need to add contributions to W.
                
                # Let's define the 7 Strassen steps explicitly.
                
                # Step 1: M1 = (A00+A11) * (B00+B11)
                # U[A00]=1, U[A11]=1
                # V[B00]=1, V[B11]=1
                # W[C00]+=1, W[C11]+=1
                
                # Global indices for A block I,K:
                # A00: (I*2+0, K*2+0)
                # A11: (I*2+1, K*2+1)
                # A01: (I*2+0, K*2+1)
                # A10: (I*2+1, K*2+0)
                
                # Global indices for B block K,J:
                # B00: (K*2+0, J*2+0)
                # B11: (K*2+1, J*2+1)
                # B01: (K*2+0, J*2+1)
                # B10: (K*2+1, J*2+0)
                
                # Global indices for C block I,J:
                # C00: (I*2+0, J*2+0)
                # C11: (I*2+1, J*2+1)
                # C01: (I*2+0, J*2+1)
                # C10: (I*2+1, J*2+0)
                
                # Helper to add to matrices
                def add_u(row, val):
                    U[row][col] += val
                def add_v(row, val):
                    V[row][col] += val
                def add_w(row, val):
                    W[row][col] += val

                # 1. M1 = (A00+A11)(B00+B11)
                add_u(I*2+0, 1); add_u(I*2+1, 1)
                add_v(K*2+0, 1); add_v(K*2+1, 1)
                add_w(I*2+0, 1); add_w(I*2+1, 1) # C00, C11
                col += 1

                # 2. M2 = (A10+A11)B00
                add_u(I*2+1, 1); add_u(I*2+1, 1) # A10, A11 -> Row I*2+1 has A10 at col 0?
                # Wait, A block is 2x2.
                # A[I*2+0, K*2+0] is A00.
                # A[I*2+0, K*2+1] is A01.
                # A[I*2+1, K*2+0] is A10.
                # A[I*2+1, K*2+1] is A11.
                
                # M2 = (A10+A11) * B00
                # A10: (I*2+1, K*2+0). A11: (I*2+1, K*2+1).
                # In the flattened U matrix, the row index is `row * 6 + col`?
                # No, U is (n*m, rank) = (36, rank).
                # Row index for A element (r, c) is r*6 + c.
                
                # Let's re-evaluate row indices.
                # U row for A[r][c] is r*6 + c.
                
                # M2 = (A10+A11)B00
                add_u((I*2+1)*6 + (K*2+0), 1) # A10
                add_u((I*2+1)*6 + (K*2+1), 1) # A11
                add_v((K*2+0)*6 + (J*2+0), 1) # B00
                # W contributions: C10 += M2, C11 += M2?
                # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
                # So C10 gets +1, C11 gets -1.
                add_w((I*2+1)*6 + (J*2+0), 1) # C10
                add_w((I*2+1)*6 + (J*2+1), -1) # C11
                col += 1

                # 3. M3 = A00(B01-B11)
                add_u((I*2+0)*6 + (K*2+0), 1) # A00
                add_v((K*2+0)*6 + (J*2+1), 1) # B01
                add_v((K*2+1)*6 + (J*2+1), -1) # B11
                # W: C01 += M3, C11 += M3
                add_w((I*2+0)*6 + (J*2+1), 1) # C01
                add_w((I*2+1)*6 + (J*2+1), 1) # C11
                col += 1

                # 4. M4 = A11(B10-B00)
                add_u((I*2+1)*6 + (K*2+1), 1) # A11
                add_v((K*2+1)*6 + (J*2+0), 1) # B10
                add_v((K*2+0)*6 + (J*2+0), -1) # B00
                # W: C00 += M4, C10 += M4
                add_w((I*2+0)*6 + (J*2+0), 1) # C00
                add_w((I*2+1)*6 + (J*2+0), 1) # C10
                col += 1

                # 5. M5 = (A00+A01)B11
                add_u((I*2+0)*6 + (K*2+0), 1) # A00
                add_u((I*2+0)*6 + (K*2+1), 1) # A01
                add_v((K*2+1)*6 + (J*2+1), 1) # B11
                # W: C00 -= M5, C01 += M5
                add_w((I*2+0)*6 + (J*2+0), -1) # C00
                add_w((I*2+0)*6 + (J*2+1), 1) # C01
                col += 1

                # 6. M6 = (A10-A00)(B00+B01)
                add_u((I*2+1)*6 + (K*2+0), 1) # A10
                add_u((I*2+0)*6 + (K*2+0), -1) # A00
                add_v((K*2+0)*6 + (J*2+0), 1) # B00
                add_v((K*2+0)*6 + (J*2+1), 1) # B01
                # W: C11 += M6
                add_w((I*2+1)*6 + (J*2+1), 1) # C11
                col += 1

                # 7. M7 = (A01-A11)(B10+B11)
                add_u((I*2+0)*6 + (K*2+1), 1) # A01
                add_u((I*2+1)*6 + (K*2+1), -1) # A11
                add_v((K*2+1)*6 + (J*2+0), 1) # B10
                add_v((K*2+1)*6 + (J*2+1), 1) # B11
                # W: C00 += M7
                add_w((I*2+0)*6 + (J*2+0), 1) # C00
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
