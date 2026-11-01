"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrices (Rank 7)
    # Indices: i, j, k in {0, 1}
    # U indices: i*2 + j (0..3)
    # V indices: j*2 + k (0..3)
    # W indices: k*2 + i (0..3)
    # Rank: 7
    
    # The 7 rank-1 terms for Strassen 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # U2: 4x7 matrix. U2[i,j, r] is 1 if the term r includes A[i,j]
    # V2: 4x7 matrix. V2[j,k, r] is 1 if the term r includes B[j,k]
    # W2: 4x7 matrix. W2[k,i, r] is 1 if the term r contributes to C[k,i]
    
    # Let's define the coefficients for each rank term r in 0..6
    # Each term r is defined by which (i,j) in A, (j,k) in B, and (k,i) in C are involved with what sign.
    
    # Term 0: (A00 + A11) * (B00 + B11)
    # A involved: (0,0), (1,1) -> U2[0,0]=1, U2[3,0]=1
    # B involved: (0,0), (0,1), (1,0), (1,1) -> V2[0,0]=1, V2[1,0]=1, V2[2,0]=1, V2[3,0]=1
    # C involved: C00 += M0, C01 += M0, C10 += M0, C11 += M0
    # Wait, M0 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # C00 = A00B00 + A01B10 + A10B00 + A11B10. 
    # This direct expansion is tricky. Let's use the standard Strassen formulas for C.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # So we define the matrices U2, V2, W2 based on these formulas.
    
    # r=0 (M1): (A00+A11)(B00+B11)
    # U2 cols: A00(1), A11(1)
    # V2 cols: B00(1), B11(1)
    # W2 cols: C00(+1), C01(0), C10(0), C11(+1) -> Wait, W2 index is k*2+i.
    # C00 is k=0,i=0 -> idx 0. C11 is k=1,i=1 -> idx 3.
    # W2[0,0]=1, W2[3,0]=1.
    
    # Let's list them explicitly to be safe.
    # r=0 (M1):
    U2_vals = [
        [1, 0, 0, 0], # A00
        [0, 0, 0, 0], # A01
        [0, 0, 0, 0], # A10
        [1, 0, 0, 0]  # A11
    ]
    # This is getting messy to hardcode row by row. Let's use the formulaic approach.
    
    # We will build U4, V4, W4 directly using the tensor product property.
    # Rank R = 7 * 7 = 49.
    # U4 shape: (16, 49)
    # V4 shape: (16, 49)
    # W4 shape: (16, 49)
    
    # First, define the 2x2 Strassen factors as lists of lists.
    # U2[i*2+j, r]
    # V2[j*2+k, r]
    # W2[k*2+i, r]
    
    # r=0 (M1): (A00+A11)(B00+B11)
    # A: 00, 11
    # B: 00, 11
    # C: 00, 11 (with signs +1, +1)
    # Note: M1 contributes to C00 and C11.
    # Formula: C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # So W2 entries:
    # r=0 (M1): C00(+1), C11(+1) -> W2[0,0]=1, W2[3,0]=1
    # r=1 (M2): (A10+A11)B00. C10(+1), C11(-1) -> W2[2,1]=1, W2[3,1]=-1
    # r=2 (M3): A00(B10-B11). C01(+1), C11(+1) -> W2[1,2]=1, W2[3,2]=1
    # r=3 (M4): A11(B01-B00). C00(-1), C10(+1) -> W2[0,3]=-1, W2[2,3]=1
    # r=4 (M5): (A00+A01)B11. C00(+1), C01(+1) -> W2[0,4]=1, W2[1,4]=1
    # r=5 (M6): (A10-A00)(B00+B10). C11(-1) -> W2[3,5]=-1
    # r=6 (M7): (A01-A11)(B01+B11). C00(-1) -> W2[0,6]=-1
    
    # Now U2 and V2.
    # U2[i*2+j, r]
    # r=0: A00(1), A11(1) -> U2[0,0]=1, U2[3,0]=1
    # r=1: A10(1), A11(1) -> U2[2,1]=1, U2[3,1]=1
    # r=2: A00(1) -> U2[0,2]=1
    # r=3: A11(1) -> U2[3,3]=1
    # r=4: A00(1), A01(1) -> U2[0,4]=1, U2[1,4]=1
    # r=5: A10(1), A00(-1) -> U2[2,5]=1, U2[0,5]=-1
    # r=6: A01(1), A11(-1) -> U2[1,6]=1, U2[3,6]=-1
    
    # V2[j*2+k, r]
    # r=0: B00(1), B11(1) -> V2[0,0]=1, V2[3,0]=1
    # r=1: B00(1) -> V2[0,1]=1
    # r=2: B10(1), B11(-1) -> V2[2,2]=1, V2[3,2]=-1
    # r=3: B01(1), B00(-1) -> V2[1,3]=1, V2[0,3]=-1
    # r=4: B11(1) -> V2[3,4]=1
    # r=5: B00(1), B10(1) -> V2[0,5]=1, V2[2,5]=1
    # r=6: B01(1), B11(1) -> V2[1,6]=1, V2[3,6]=1
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Fill U2
    U2[0][0] = 1; U2[3][0] = 1
    U2[2][1] = 1; U2[3][1] = 1
    U2[0][2] = 1
    U2[3][3] = 1
    U2[0][4] = 1; U2[1][4] = 1
    U2[2][5] = 1; U2[0][5] = -1
    U2[1][6] = 1; U2[3][6] = -1
    
    # Fill V2
    V2[0][0] = 1; V2[3][0] = 1
    V2[0][1] = 1
    V2[2][2] = 1; V2[3][2] = -1
    V2[1][3] = 1; V2[0][3] = -1
    V2[3][4] = 1
    V2[0][5] = 1; V2[2][5] = 1
    V2[1][6] = 1; V2[3][6] = 1
    
    # Fill W2
    W2[0][0] = 1; W2[3][0] = 1
    W2[2][1] = 1; W2[3][1] = -1
    W2[1][2] = 1; W2[3][2] = 1
    W2[0][3] = -1; W2[2][3] = 1
    W2[0][4] = 1; W2[1][4] = 1
    W2[3][5] = -1
    W2[0][6] = -1
    
    # Now construct 4x4 factors using Kronecker product logic.
    # Rank R = 49
    # U4[16, 49], V4[16, 49], W4[16, 49]
    
    R = 49
    U4 = [[0]*R for _ in range(16)]
    V4 = [[0]*R for _ in range(16)]
    W4 = [[0]*R for _ in range(16)]
    
    # Mapping 4x4 indices to 2x2 block/sub indices
    # i = 2*i1 + i2
    # j = 2*j1 + j2
    # k = 2*k1 + k2
    
    # For U4[i*4+j, r1*7+r2]:
    # i_block = 2*i1 + j1
    # i_sub = 2*i2 + j2
    # U4[i*4+j, r1*7+r2] = U2[i_block, r1] * U2[i_sub, r2]
    
    # For V4[j*4+k, r1*7+r2]:
    # j_block = 2*j1 + k1
    # j_sub = 2*j2 + k2
    # V4[j*4+k, r1*7+r2] = V2[j_block, r1] * V2[j_sub, r2]
    
    # For W4[k*4+i, r1*7+r2]:
    # k_block = 2*k1 + i1
    # k_sub = 2*k2 + i2
    # W4[k*4+i, r1*7+r2] = W2[k_block, r1] * W2[k_sub, r2]
    
    # Precompute index mappings
    # a = i*4+j
    # i = a // 4
    # j = a % 4
    # i1 = i // 2, i2 = i % 2
    # j1 = j // 2, j2 = j % 2
    # i_block = 2*i1 + j1
    # i_sub = 2*i2 + j2
    
    # b = j*4+k
    # j = b // 4
    # k = b % 4
    # j1 = j // 2, j2 = j % 2
    # k1 = k // 2, k2 = k % 2
    # j_block = 2*j1 + k1
    # j_sub = 2*j2 + k2
    
    # c = k*4+i
    # k = c // 4
    # i = c % 4
    # k1 = k // 2, k2 = k % 2
    # i1 = i // 2, i2 = i % 2
    # k_block = 2*k1 + i1
    # k_sub = 2*k2 + i2
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for a in range(16):
                i = a // 4
                j = a % 4
                i1 = i // 2
                i2 = i % 2
                j1 = j // 2
                j2 = j % 2
                i_block = 2 * i1 + j1
                i_sub = 2 * i2 + j2
                U4[a][r] = U2[i_block][r1] * U2[i_sub][r2]
                
            for b in range(16):
                j = b // 4
                k = b % 4
                j1 = j // 2
                j2 = j % 2
                k1 = k // 2
                k2 = k % 2
                j_block = 2 * j1 + k1
                j_sub = 2 * j2 + k2
                V4[b][r] = V2[j_block][r1] * V2[j_sub][r2]
                
            for c in range(16):
                k = c // 4
                i = c % 4
                k1 = k // 2
                k2 = k % 2
                i1 = i // 2
                i2 = i % 2
                k_block = 2 * k1 + i1
                k_sub = 2 * k2 + i2
                W4[c][r] = W2[k_block][r1] * W2[k_sub][r2]
                
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
