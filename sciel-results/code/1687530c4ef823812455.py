import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    
    # We will build the decomposition using block Strassen for the 2x2 part of A and B.
    # A is 2x3. Split into A1 (2x2) and A2 (2x1).
    # B is 3x9. Split into B1 (2x9) and B2 (1x9).
    # C = A1 B1 + A2 B2
    
    # A1 B1 is 2x2 times 2x9.
    # Split B1 into 4 blocks of 2x2 (cols 0-1, 2-3, 4-5, 6-7) and 1 block of 2x1 (col 8).
    # B1_1 (2x2), B1_2 (2x2), B1_3 (2x2), B1_4 (2x2), B1_5 (2x1).
    
    # Rank contributions:
    # 4 * Rank(2x2x2) + Rank(2x2x1) + Rank(2x1x9)
    # Rank(2x2x2) = 7 (Strassen)
    # Rank(2x2x1) = 4 (Schoolbook)
    # Rank(2x1x9) = 18 (Schoolbook)
    # Total Rank = 4*7 + 4 + 18 = 28 + 4 + 18 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)] # 6 x 50
    V = [[0] * rank for _ in range(m * p)] # 27 x 50
    W = [[0] * rank for _ in range(p * n)] # 18 x 50
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M2 + M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M5 - M7
    
    # We need to map these to U, V, W.
    # U has indices for A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # V has indices for B: j*p + k.
    #   B1 is rows 0,1.
    #   B1_1: rows 0,1, cols 0,1. Indices: 0*9+0=0, 0*9+1=1, 1*9+0=9, 1*9+1=10.
    #   B1_2: rows 0,1, cols 2,3. Indices: 2,3,11,12.
    #   ...
    # W has indices for C: k*n + i.
    #   C is 9x2.
    #   Block 1 (cols 0,1 of B1) -> C cols 0,1.
    #   C indices: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3.
    
    def strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # a00, a01, a10, a11 are indices in U
        # b00, b01, b10, b11 are indices in V
        # c00, c01, c10, c11 are indices in W
        
        # M1 = (a00+a11)(b00+b11)
        # We need to represent (a00+a11) in U and (b00+b11) in V?
        # No, the decomposition is:
        # C = sum_r U[:,r] * V[:,r] * W[:,r]
        # Each term r corresponds to a product of a linear form in A and a linear form in B.
        # So for M1, U has 1 at a00 and a11. V has 1 at b00 and b11. W has coeffs.
        
        # M1
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M2 = (a10+a11)(b00+b11)
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[c10][col] = 1
        W[c00][col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[a00][col] = 1
        V[b01][col] = 1
        V[b11][col] = -1
        W[c01][col] = 1
        W[c11][col] = -1
        col += 1
        
        # M4 = a11(b10-b00)
        U[a11][col] = 1
        V[b10][col] = 1
        V[b00][col] = -1
        W[c00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        W[c01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00][col] = 1
        V[b01][col] = 1
        W[c10][col] = 1
        W[c00][col] = -1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][col] = 1
        U[a11][col] = -1
        V[b10][col] = 1
        V[b11][col] = 1
        W[c11][col] = 1
        W[c00][col] = 1
        col += 1
        
        return col

    # Indices for A1 (2x2 part of A, cols 0,1)
    # A row 0: cols 0,1 -> U[0], U[1]
    # A row 1: cols 0,1 -> U[3], U[4]
    a00, a01, a10, a11 = 0, 1, 3, 4
    
    # Process 4 blocks of 2x2 in B1
    for block_idx in range(4):
        # B1 block cols: block_idx*2, block_idx*2+1
        # V indices:
        # Row 0: 0*9 + block_idx*2, 0*9 + block_idx*2+1
        # Row 1: 1*9 + block_idx*2, 1*9 + block_idx*2+1
        b00 = 0 * 9 + block_idx * 2
        b01 = 0 * 9 + block_idx * 2 + 1
        b10 = 1 * 9 + block_idx * 2
        b11 = 1 * 9 + block_idx * 2 + 1
        
        # W indices for C block (cols block_idx*2, block_idx*2+1)
        # C col 0: 0*2+0=0, 0*2+1=1? No.
        # C is 9x2. Index k*n + i.
        # Block 0: C cols 0,1.
        # C[0,0] -> 0*2+0=0
        # C[0,1] -> 0*2+1=1
        # C[1,0] -> 1*2+0=2
        # C[1,1] -> 1*2+1=3
        # Block 1: C cols 2,3.
        # C[2,0] -> 2*2+0=4
        # C[2,1] -> 2*2+1=5
        # C[3,0] -> 3*2+0=6
        # C[3,1] -> 3*2+1=7
        
        c00 = block_idx * 2 * 2 + 0
        c01 = block_idx * 2 * 2 + 1
        c10 = block_idx * 2 * 2 + 2
        c11 = block_idx * 2 * 2 + 3
        
        strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
        
    # Process 1 block of 2x1 in B1 (col 8)
    # V indices:
    b00 = 0 * 9 + 8
    b10 = 1 * 9 + 8
    
    # W indices for C col 8
    c00 = 8 * 2 + 0
    c10 = 8 * 2 + 1
    
    # Schoolbook 2x2x1
    # C00 = a00*b00 + a01*b10
    # C10 = a10*b00 + a11*b10
    
    # Term 1: a00 b00
    U[a00][col] = 1
    V[b00][col] = 1
    W[c00][col] = 1
    col += 1
    
    # Term 2: a01 b10
    U[a01][col] = 1
    V[b10][col] = 1
    W[c00][col] = 1
    col += 1
    
    # Term 3: a10 b00
    U[a10][col] = 1
    V[b00][col] = 1
    W[c10][col] = 1
    col += 1
    
    # Term 4: a11 b10
    U[a11][col] = 1
    V[b10][col] = 1
    W[c10][col] = 1
    col += 1
    
    # Process A2 B2
    # A2 is 2x1 (col 2 of A). Indices: U[2] (row 0), U[5] (row 1).
    # B2 is 1x9 (row 2 of B). Indices: V[2*9 + k] for k=0..8.
    # C is 9x2.
    # C[i, k] += A[i, 2] B[2, k]
    # W index: k*2 + i.
    
    for k in range(9):
        v_idx = 2 * 9 + k
        w_0 = k * 2 + 0
        w_1 = k * 2 + 1
        
        # Term 1: A[0,2] B[2,k]
        U[2][col] = 1
        V[v_idx][col] = 1
        W[w_0][col] = 1
        col += 1
        
        # Term 2: A[1,2] B[2,k]
        U[5][col] = 1
        V[v_idx][col] = 1
        W[w_1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
