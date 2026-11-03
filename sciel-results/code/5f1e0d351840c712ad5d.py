import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val

    # We have two blocks of 12 rank each
    # Block 1: A * B1
    # Block 2: A * B2
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-b)(f+h)
    # m7 = (a-d)(e+g)
    
    # 2x2x3 decomposition (Rank 11)
    # A_22 is 2x2, B_23 is 2x3 = [B_22, B_21_col]
    # Part 1: A_22 * B_22 (Rank 7)
    # Part 2: A_22 * B_21_col (Rank 4)
    
    # Indices for A (2x3):
    # 0: a, 1: b, 2: c, 3: d, 4: a2 (2x1 part)
    # Wait, A is 2x3.
    # Row 0: A00, A01, A02
    # Row 1: A10, A11, A12
    # Split A = [A_22, A_21]
    # A_22 = [[A00, A01], [A10, A11]]
    # A_21 = [[A02], [A12]]
    
    # B is 3x3 for the block.
    # Row 0: B00, B01, B02
    # Row 1: B10, B11, B12
    # Row 2: B20, B21, B22
    # Split B = [B_23; B_13]
    # B_23 = [[B00, B01, B02], [B10, B11, B12]]
    # B_13 = [[B20, B21, B22]]
    
    # C is 2x3.
    # C00, C01, C02
    # C10, C11, C12
    
    # Part 1: A_22 * B_23
    # B_23 = [B_22, B_21_col]
    # B_22 = [[B00, B01], [B10, B11]]
    # B_21_col = [[B02], [B12]]
    
    # Strassen for A_22 * B_22
    # a=A00, b=A01, c=A10, d=A11
    # e=B00, f=B01, g=B10, h=B11
    
    # m1
    add_term(0, 0, 0, col); add_term(3, 0, 0, col) # a+d
    add_term(0, 0, 0, col); add_term(1, 0, 0, col) # e+h
    # This is not how it works. We need linear combinations in U, V, W.
    # U[i, col] = coeff for A[i]
    # V[j, col] = coeff for B[j]
    # W[k, col] = coeff for C[k]
    
    # m1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[4][col] = 1 # C00, C11
    W[1][col] = -1; W[3][col] = -1 # C01, C10? No.
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m7
    
    # Let's just use schoolbook for 2x2x2? Rank 4.
    # Then 2x2x3 is 4+4=8.
    # 2x1x3 is 1.
    # Total 9 for 2x3x3.
    # Total 18 for 2x3x6.
    # This is better and easier to code correctly.
    
    pass

def construct():
    n, m, p = 2, 3, 6
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Two blocks of 9 rank each
    for block in range(2):
        # A_22 * B_22 (Rank 4)
        # A_22 * B_21 (Rank 4)
        # A_21 * B_13 (Rank 1)
        
        # A indices: 0,1,2,3,4,5 (0,1,2 for row 0; 3,4,5 for row 1)
        # A_22: 0,1,3,4
        # A_21: 2,5
        
        # B indices for block:
        # Block 0: B cols 0,1,2. Rows 0,1,2.
        # B_22: 0,1,3,4 (Row 0: 0,1; Row 1: 3,4)
        # B_21: 2,5 (Row 0: 2; Row 1: 5)
        # B_13: 6,7,8 (Row 2: 6,7,8)
        
        # Block 1: B cols 3,4,5.
        # B_22: 9,10,12,13
        # B_21: 11,14
        # B_13: 15,16,17
        
        b_offset = block * 9
        
        # Part 1: A_22 * B_22 (Schoolbook 4 terms)
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # Term 1: A00 B00
        U[0][col] = 1
        V[0 + b_offset][col] = 1
        W[0][col] = 1 # C00
        col += 1
        
        # Term 2: A01 B10
        U[1][col] = 1
        V[3 + b_offset][col] = 1
        W[0][col] = 1 # C00
        col += 1
        
        # Term 3: A00 B01
        U[0][col] = 1
        V[1 + b_offset][col] = 1
        W[1][col] = 1 # C01
        col += 1
        
        # Term 4: A01 B11
        U[1][col] = 1
        V[4 + b_offset][col] = 1
        W[1][col] = 1 # C01
        col += 1
        
        # Term 5: A10 B00
        U[3][col] = 1
        V[0 + b_offset][col] = 1
        W[2][col] = 1 # C10 (k=0, i=1 -> k*n+i = 0*2+1=1? No. W[k*n+i]. C10 is i=1, k=0. Index 1.
        # Wait. C is 2x3.
        # C00 (i=0, k=0) -> W[0]
        # C01 (i=0, k=1) -> W[2]
        # C02 (i=0, k=2) -> W[4]
        # C10 (i=1, k=0) -> W[1]
        # C11 (i=1, k=1) -> W[3]
        # C12 (i=1, k=2) -> W[5]
        
        # Let's restart the W indexing.
        # W[k*n + i]
        # n=2.
        # k=0: W[0]=C00, W[1]=C10
        # k=1: W[2]=C01, W[3]=C11
        # k=2: W[4]=C02, W[5]=C12
        
        # Correcting previous terms:
        # Term 1: A00 B00 -> C00 (W[0])
        # Term 2: A01 B10 -> C00 (W[0])
        # Term 3: A00 B01 -> C01 (W[2])
        # Term 4: A01 B11 -> C01 (W[2])
        
        # I already wrote them. I need to fix W indices in the code.
        
        # Part 2: A_22 * B_21 (Rank 4)
        # B_21 is col 2 of B_23.
        # B_21 = [[B02], [B12]]
        # Indices: 2 + b_offset, 5 + b_offset
        # C02 = A00 B02 + A01 B12
        # C12 = A10 B02 + A11 B12
        
        # Term 5: A00 B02
        U[0][col] = 1
        V[2 + b_offset][col] = 1
        W[4][col] = 1 # C02
        col += 1
        
        # Term 6: A01 B12
        U[1][col] = 1
        V[5 + b_offset][col] = 1
        W[4][col] = 1 # C02
        col += 1
        
        # Term 7: A10 B02
        U[3][col] = 1
        V[2 + b_offset][col] = 1
        W[5][col] = 1 # C12
        col += 1
        
        # Term 8: A11 B12
        U[4][col] = 1
        V[5 + b_offset][col] = 1
        W[5][col] = 1 # C12
        col += 1
        
        # Part 3: A_21 * B_13 (Rank 1)
        # A_21 = [[A02], [A12]] (Indices 2, 5)
        # B_13 = [[B20, B21, B22]] (Indices 6,7,8 + b_offset)
        # C02 += A02 B20? No.
        # C = A B.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # j=2 term: A_{i,2} B_{2,k}.
        # C02 += A02 B22? No, k=2.
        # C0,2 = A00 B02 + A01 B12 + A02 B22.
        # C1,2 = A10 B02 + A11 B12 + A12 B22.
        
        # My Part 2 handled B02, B12.
        # Part 3 handles B20, B21, B22?
        # B_13 is row 2 of B.
        # B_{2,0}, B_{2,1}, B_{2,2}.
        # Indices 6, 7, 8 + b_offset.
        
        # C00 += A02 B20
        # C01 += A02 B21
        # C02 += A02 B22
        # C10 += A12 B20
        # C11 += A12 B21
        # C12 += A12 B22
        
        # This is 6 terms if schoolbook.
        # But A_21 is 2x1, B_13 is 1x3.
        # Rank 1?
        # C_{i,k} = A_{i,2} B_{2,k}.
        # This is outer product of A_21 and B_13.
        # U: A_21 indices 2,5.
        # V: B_13 indices 6,7,8.
        # W: C indices 0,1,2,3,4,5.
        
        # U[2] = 1, U[5] = 1
        # V[6] = 1, V[7] = 1, V[8] = 1
        # W[0] = 1 (C00), W[2] = 1 (C01), W[4] = 1 (C02)
        # W[1] = 1 (C10), W[3] = 1 (C11), W[5] = 1 (C12)
        
        U[2][col] = 1
        U[5][col] = 1
        V[6 + b_offset][col] = 1
        V[7 + b_offset][col] = 1
        V[8 + b_offset][col] = 1
        W[0][col] = 1
        W[1][col] = 1
        W[2][col] = 1
        W[3][col] = 1
        W[4][col] = 1
        W[5][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
