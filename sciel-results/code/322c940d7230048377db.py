import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition: 4 blocks of 2x2x2 Strassen
    # Each 2x2x2 block has rank 7.
    # Total rank 4 * 7 = 28? 
    # Wait. 
    # My previous analysis: 
    # R(2, 4, 8) <= 4 * R(2, 2, 4).
    # R(2, 2, 4) <= 2 * R(2, 2, 2) = 14.
    # So 4 * 14 = 56.
    
    # Let's verify the block structure again.
    # A (2x4) = [A1 A2] (2x2 each)
    # B (4x8) = [[B11 B12] [B21 B22]] (2x4 each)
    # C (2x8) = [C1 C2] (2x4 each)
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # C1 is sum of two 2x2x4 multiplications?
    # A1 (2x2) * B11 (2x4) -> 2x4 result.
    # This is M_2,2,4.
    # Rank of M_2,2,4 is 14 (using 2x Strassen).
    # So C1 requires 14 + 14 = 28 products?
    # No, C1 = A1 B11 + A2 B21.
    # These two terms are disjoint in A and B indices.
    # So Rank(C1) = Rank(A1 B11) + Rank(A2 B21) = 14 + 14 = 28.
    # Similarly C2 requires 28 products.
    # Total 56.
    
    # So I need to implement 4 instances of M_2,2,4.
    # Each M_2,2,4 is implemented as 2 instances of M_2,2,2 (Strassen).
    # So total 8 instances of Strassen.
    # 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 2x2x2 coefficients
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have 8 Strassen blocks.
    # Block 1: A1, B11 -> C1_part1
    # Block 2: A2, B21 -> C1_part2
    # Block 3: A1, B12 -> C2_part1
    # Block 4: A2, B22 -> C2_part2
    
    # Each block is 2x2x4? No.
    # A1 is 2x2. B11 is 2x4.
    # So Block 1 is M_2,2,4.
    # M_2,2,4 is split into two M_2,2,2.
    # B11 = [B11a B11b] (2x2 each).
    # C1_part1 = A1 B11a + A1 B11b.
    # So Block 1 consists of 2 Strassen multiplications.
    # Strassen 1.1: A1, B11a -> C11a
    # Strassen 1.2: A1, B11b -> C11b
    # C11a and C11b are 2x2 matrices.
    # C1_part1 (2x4) = [C11a C11b].
    
    # So we have 8 Strassen multiplications total.
    # 1.1, 1.2 (for A1, B11)
    # 2.1, 2.2 (for A2, B21)
    # 3.1, 3.2 (for A1, B12)
    # 4.1, 4.2 (for A2, B22)
    
    # Indices for U (A): 2x4 = 8 entries.
    # A1 is cols 0,1. A2 is cols 2,3.
    # A1: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1].
    # A2: a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3].
    
    # Indices for V (B): 4x8 = 32 entries.
    # B11 is rows 0,1 and cols 0,1.
    # B11a: B[0:2, 0:2]. B11b: B[0:2, 2:4].
    # B21 is rows 2,3 and cols 0,1.
    # B21a: B[2:4, 0:2]. B21b: B[2:4, 2:4].
    # B12 is rows 0,1 and cols 4,5,6,7?
    # Wait, B is 4x8.
    # B11: rows 0,1; cols 0,1,2,3. (2x4)
    # B12: rows 0,1; cols 4,5,6,7. (2x4)
    # B21: rows 2,3; cols 0,1,2,3. (2x4)
    # B22: rows 2,3; cols 4,5,6,7. (2x4)
    
    # So:
    # Block 1 (A1, B11):
    #   Sub 1.1: A1, B[0:2, 0:2]
    #   Sub 1.2: A1, B[0:2, 2:4]
    # Block 2 (A2, B21):
    #   Sub 2.1: A2, B[2:4, 0:2]
    #   Sub 2.2: A2, B[2:4, 2:4]
    # Block 3 (A1, B12):
    #   Sub 3.1: A1, B[0:2, 4:6]
    #   Sub 3.2: A1, B[0:2, 6:8]
    # Block 4 (A2, B22):
    #   Sub 4.1: A2, B[2:4, 4:6]
    #   Sub 4.2: A2, B[2:4, 6:8]
    
    # Output C (2x8).
    # C1 (cols 0-3) = A1 B11 + A2 B21.
    # C1 = [C11a C11b C21a C21b]?
    # No.
    # A1 B11 (2x4) + A2 B21 (2x4) = C1 (2x4).
    # A1 B11 = [A1 B11a, A1 B11b].
    # A2 B21 = [A2 B21a, A2 B21b].
    # So C1 = [A1 B11a + A2 B21a, A1 B11b + A2 B21b].
    # C2 (cols 4-7) = A1 B12 + A2 B22.
    # C2 = [A1 B12a + A2 B22a, A1 B12b + A2 B22b].
    
    # W indices: k*n + i.
    # C is 2x8. i in 0,1. k in 0..7.
    # W index = k*2 + i.
    
    # I will construct the 8 Strassen blocks.
    
    strassen_blocks = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        # A is 2x4. B is 4x8. C is 2x8.
        # 1.1: A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 0, 0, 0),
        # 1.2: A[0:2, 0:2], B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 0, 0, 2, 0, 2),
        # 2.1: A[0:2, 2:4], B[2:4, 0:2] -> C[0:2, 0:2] (Add to C)
        (0, 2, 2, 0, 0, 0),
        # 2.2: A[0:2, 2:4], B[2:4, 2:4] -> C[0:2, 2:4] (Add to C)
        (0, 2, 2, 2, 0, 2),
        # 3.1: A[0:2, 0:2], B[0:2, 4:6] -> C[0:2, 4:6]
        (0, 0, 0, 4, 0, 4),
        # 3.2: A[0:2, 0:2], B[0:2, 6:8] -> C[0:2, 6:8]
        (0, 0, 0, 6, 0, 6),
        # 4.1: A[0:2, 2:4], B[2:4, 4:6] -> C[0:2, 4:6] (Add to C)
        (0, 2, 2, 4, 0, 4),
        # 4.2: A[0:2, 2:4], B[2:4, 6:8] -> C[0:2, 6:8] (Add to C)
        (0, 2, 2, 6, 0, 6),
    ]
    
    # Strassen coefficients for U, V, W
    # U: (a,d,c,d,a,a,d,b,c,a,b,b,c,b,d) -> indices 0,1,2,3
    # V: (e,h,e,e,f,h,h,e,f,f,g,h,g,g) -> indices 0,1,2,3
    # W: (1,-1,0,0, 1,1,0,0, 0,1,1,0, 0,0,1,1) ?
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1: U=(1,0,0,1), V=(1,0,0,1)
    # M2: U=(0,0,1,1), V=(1,0,0,0)
    # M3: U=(1,0,0,0), V=(0,1,0,-1)
    # M4: U=(0,0,0,1), V=(0,0,1,-1)
    # M5: U=(1,1,0,0), V=(0,0,0,1)
    # M6: U=(0,-1,1,0), V=(1,1,0,0)
    # M7: U=(0,1,-1,0), V=(0,0,1,1)
    
    # W coefficients:
    # M1: (1, 0, 0, -1)
    # M2: (0, 0, 1, -1)
    # M3: (0, 1, 0, 1)
    # M4: (-1, 0, 1, 0)
    # M5: (1, 0, 0, 0)
    # M6: (0, 0, 0, 1)
    # M7: (-1, 1, 0, 0)
    
    # Note: W index for C is k*n + i.
    # For a 2x2 block at C_row_offset, C_col_offset:
    # C[i, k] is at W[k*2 + i].
    # In the block, local i, k are 0,1.
    # Global i = C_row_offset + i_local.
    # Global k = C_col_offset + k_local.
    # W index = (C_col_offset + k_local)*2 + (C_row_offset + i_local).
    
    col = 0
    for (ar, ac, br, bc, cr, cc) in strassen_blocks:
        for m_idx in range(7):
            # U
            if m_idx == 0: U[ar*2+ac+0][col] += 1; U[ar*2+ac+3][col] += 1
            elif m_idx == 1: U[ar*2+ac+2][col] += 1; U[ar*2+ac+3][col] += 1
            elif m_idx == 2: U[ar*2+ac+0][col] += 1
            elif m_idx == 3: U[ar*2+ac+3][col] += 1
            elif m_idx == 4: U[ar*2+ac+0][col] += 1; U[ar*2+ac+1][col] += 1
            elif m_idx == 5: U[ar*2+ac+1][col] -= 1; U[ar*2+ac+2][col] += 1
            elif m_idx == 6: U[ar*2+ac+1][col] += 1; U[ar*2+ac+3][col] -= 1
            
            # V
            if m_idx == 0: V[br*2+bc+0][col] += 1; V[br*2+bc+3][col] += 1
            elif m_idx == 1: V[br*2+bc+0][col] += 1
            elif m_idx == 2: V[br*2+bc+1][col] += 1; V[br*2+bc+3][col] -= 1
            elif m_idx == 3: V[br*2+bc+2][col] += 1; V[br*2+bc+0][col] -= 1
            elif m_idx == 4: V[br*2+bc+3][col] += 1
            elif m_idx == 5: V[br*2+bc+0][col] += 1; V[br*2+bc+1][col] += 1
            elif m_idx == 6: V[br*2+bc+2][col] += 1; V[br*2+bc+3][col] += 1
            
            # W
            # M1: C11+=1, C22-=1
            # M2: C21+=1, C22-=1
            # M3: C12+=1, C22+=1
            # M4: C11-=1, C21+=1
            # M5: C11+=1
            # M6: C22+=1
            # M7: C11-=1, C12+=1
            
            def add_w(i_loc, k_loc, val):
                i_g = cr + i_loc
                k_g = cc + k_loc
                idx = k_g * n + i_g
                W[idx][col] += val
            
            if m_idx == 0: add_w(0, 0, 1); add_w(1, 1, -1)
            elif m_idx == 1: add_w(1, 0, 1); add_w(1, 1, -1)
            elif m_idx == 2: add_w(0, 1, 1); add_w(1, 1, 1)
            elif m_idx == 3: add_w(0, 0, -1); add_w(1, 0, 1)
            elif m_idx == 4: add_w(0, 0, 1)
            elif m_idx == 5: add_w(1, 1, 1)
            elif m_idx == 6: add_w(0, 0, -1); add_w(0, 1, 1)
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
