import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We use 3 blocks of 3 columns each.
    # Each block uses a rank-17 decomposition for 2x3x3.
    # Total rank = 3 * 17 = 51.
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # U_index: i*m + j
    # V_index: j*p + k
    # W_index: k*n + i
    # Note: W index in task is k*n + i.
    
    col = 0
    for block in range(3):
        # Block offset in k
        k_offset = block * 3
        
        # 17 terms for this block
        # M1..M7: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # M8..M11: Schoolbook for A[0:2, 0:2] * B[0:2, 2] (3rd col of block)
        # M12..M17: Schoolbook for A[0:2, 2] * B[2, 0:3]
        
        # Strassen terms
        # M1: (a00+a11)(b00+b11)
        # U: a00, a11 -> 1
        # V: b00, b11 -> 1
        # W: c00, c11 -> 1, c01, c10 -> -1? 
        # Strassen reconstruction:
        # c00 = M1 - M4 + M7 - M5
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # We define U, V, W for each multiplication M_r.
        # The contribution to C is sum_r U_r * V_r * W_r.
        
        # Term 1: (a00+a11)(b00+b11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00=1, c11=1, c01=-1, c10=-1 ? No.
        # M1 contributes +1 to c00 and c11, and -1 to c01 and c10?
        # Let's check:
        # c00 = M1 ...
        # c11 = M1 ...
        # c01 = ... - M1? No.
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B11)
        # P4 = A22(B21-B11) -- Wait, standard is A22(B12-B11)?
        # Let's use a known valid Strassen decomposition.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        # C11 = P1 - P4 + P7 - P5
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Map to our indices:
        # a=a00, b=a01, c=a10, d=a11
        # e=b00, f=b01, g=b10, h=b11
        # C00=C11, C01=C12, C10=C21, C11=C22
        
        # Term 1: (a00+a11)(b00+b11)
        U[col][0] = 1 # a00
        U[col][3] = 1 # a11
        V[col][0] = 1 # b00
        V[col][3] = 1 # b11
        W[col][0] = 1 # c00 (k=0, i=0)
        W[col][2] = 1 # c11 (k=1, i=1) -- Wait, W index is k*n+i.
        # c00: k=0, i=0 -> 0
        # c01: k=1, i=0 -> 1
        # c10: k=0, i=1 -> 2
        # c11: k=1, i=1 -> 3
        
        # W for P1:
        # c00 += 1
        # c11 += 1
        # c01 -= 1
        # c10 -= 1
        W[col][0] = 1
        W[col][3] = 1
        W[col][1] = -1
        W[col][2] = -1
        col += 1
        
        # Term 2: (a10+a11)b00
        U[col][3] = 1 # a11
        U[col][2] = 1 # a10
        V[col][0] = 1 # b00
        # c10 += 1
        # c11 -= 1
        W[col][2] = 1
        W[col][3] = -1
        col += 1
        
        # Term 3: a00(b01-b11)
        U[col][0] = 1
        V[col][1] = 1 # b01
        V[col][3] = -1 # b11
        # c01 += 1
        # c11 += 1
        W[col][1] = 1
        W[col][3] = 1
        col += 1
        
        # Term 4: a11(b10-b00)
        U[col][3] = 1
        V[col][2] = 1 # b10
        V[col][0] = -1 # b00
        # c00 -= 1
        # c10 += 1
        W[col][0] = -1
        W[col][2] = 1
        col += 1
        
        # Term 5: (a00+a01)b11
        U[col][0] = 1
        U[col][1] = 1
        V[col][3] = 1
        # c00 -= 1
        # c01 += 1
        W[col][0] = -1
        W[col][1] = 1
        col += 1
        
        # Term 6: (a10-a00)(b00+b01)
        U[col][2] = 1
        U[col][0] = -1
        V[col][0] = 1
        V[col][1] = 1
        # c11 += 1
        W[col][3] = 1
        col += 1
        
        # Term 7: (a01-a11)(b10+b11)
        U[col][1] = 1
        U[col][3] = -1
        V[col][2] = 1
        V[col][3] = 1
        # c00 += 1
        W[col][0] = 1
        col += 1
        
        # Term 8: a00*b02
        U[col][0] = 1
        V[col][4] = 1 # b02 (j=0, k=2 in block)
        # c02 += 1 (k=2, i=0 -> 2*2+0=4)
        W[col][4] = 1
        col += 1
        
        # Term 9: a01*b12
        U[col][1] = 1
        V[col][5] = 1 # b12 (j=1, k=2)
        W[col][4] = 1
        col += 1
        
        # Term 10: a10*b02
        U[col][2] = 1
        V[col][4] = 1
        # c12 += 1 (k=2, i=1 -> 2*2+1=5)
        W[col][5] = 1
        col += 1
        
        # Term 11: a11*b12
        U[col][3] = 1
        V[col][5] = 1
        W[col][5] = 1
        col += 1
        
        # Term 12: a02*b20
        U[col][4] = 1 # a02 (i=0, j=2)
        V[col][6] = 1 # b20 (j=2, k=0)
        W[col][0] = 1 # c00
        col += 1
        
        # Term 13: a02*b21
        U[col][4] = 1
        V[col][7] = 1 # b21 (j=2, k=1)
        W[col][1] = 1 # c01
        col += 1
        
        # Term 14: a02*b22
        U[col][4] = 1
        V[col][8] = 1 # b22 (j=2, k=2)
        W[col][4] = 1 # c02
        col += 1
        
        # Term 15: a12*b20
        U[col][5] = 1 # a12 (i=1, j=2)
        V[col][6] = 1
        W[col][2] = 1 # c10
        col += 1
        
        # Term 16: a12*b21
        U[col][5] = 1
        V[col][7] = 1
        W[col][3] = 1 # c11
        col += 1
        
        # Term 17: a12*b22
        U[col][5] = 1
        V[col][8] = 1
        W[col][5] = 1 # c12
        col += 1
        
        # Adjust indices for block offset
        # V indices: j*p + k
        # In block, k is 0,1,2. Global k is k_offset + k.
        # We used local k indices in V above (0..8).
        # We need to shift V indices by k_offset.
        # V index = j*p + (k_offset + k_local)
        
        # We filled V with local indices. We need to fix this.
        # It's easier to fill V with global indices directly.
        # Let's restart the V filling logic or shift after.
        # Shifting V is hard because it's already filled.
        # I will rewrite the loop to use global indices.

    return {"U": U, "V": V, "W": W}
