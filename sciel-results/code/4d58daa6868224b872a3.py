import time
import numpy as np

def construct():
    n, m, p = 4, 4, 9

    # We'll use a block approach:
    # Split A (4x4) into 2x2 blocks of 2x2
    # Split B (4x9) into 2x3 blocks of 2x3
    # Use Strassen-like formulas but adapted for rectangular blocks
    
    # Actually, let's use a simpler approach: 
    # Treat the 9 columns as 2 full 4x4 multiplications + 1 column
    # 4x4x4 can be done with rank 19 (using 2x2 Strassen with rank 7 per 2x2x2)
    # But let's start with a cleaner decomposition
    
    # Approach: Use the fact that we can do 4x4x4 with rank 19
    # and 4x4x1 with rank 16 (schoolbook)
    # Total: 19 + 19 + 16 = 54? No, that's not right.
    
    # Let me think more carefully.
    # The tensor for 4x4x9 multiplication has rank at most 4*4*9 = 144 (schoolbook)
    # We can do better.
    
    # Key insight: We can split B into two parts: B1 (4x4) and B2 (4x5)
    # C = A*B1 + A*B2
    # A*B1 is 4x4x4 (rank at most 19)
    # A*B2 is 4x4x5 (rank at most 20 schoolbook, or better)
    
    # Actually, let's use a known result: 
    # 4x4x4 matrix multiplication can be done in rank 19
    # This uses the 2x2 Strassen (rank 7) applied to 2x2 blocks
    # 7 multiplications of 2x2x2, each rank 7, total 49? No...
    
    # Wait, 2x2x2 has rank 7 (Strassen). 
    # For 4x4x4, we split into 2x2 blocks, getting 4 subproblems of 2x2x2
    # But Strassen reduces 8 multiplications to 7.
    # So 4x4x4 rank is at most 7*7 = 49? That's not right either.
    
    # Let me be more careful.
    # 2x2x2 tensor: rank 7
    # 4x4x4: Split A into 2x2 blocks A11,A12,A21,A22 (each 2x2)
    #         B into 2x2 blocks B11,B12,B21,B22 (each 2x2)
    #         C into 2x2 blocks C11,C12,C21,C22 (each 2x2)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Each product like A11*B11 is a 2x2x2 multiplication (rank 7)
    # So naively: 8 products * rank 7 = rank 56
    # Strassen: 7 products * rank 7 = rank 49
    # 
    # But there's a better way. The 4x4x4 tensor itself has rank 19.
    # This is a known result (Lader, 1974, etc.)
    
    # Let me construct the rank-19 decomposition for 4x4x4.
    # This is complex. Let me use a simpler but still improved approach.
    
    # Alternative: Use 2x2 Strassen for the 4x4x4 part.
    # 4x4x4 with Strassen: rank 49 (7 Strassen steps, each 2x2x2 rank 7)
    # 4x4x5: split into 4x4x4 + 4x4x1, rank 49 + 16 = 65
    # Total for 4x4x9: 49 + 49 + 16 = 114
    
    # Let's implement this.
    
    # First, define the 2x2x2 Strassen decomposition (rank 7)
    # U2, V2, W2 are 4x7 matrices
    
    # 2x2x2: A is 2x2, B is 2x2, C = A*B is 2x2
    # Indices: A[i,k] at i*2+k, B[k,j] at k*2+j, C[i,j] at j*2+i (trace convention)
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M4
    # C21 = M2 + M6
    # C22 = M1 - M2 - M5 + M7
    
    # Let's build U2, V2, W2 for 2x2x2
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3 (W convention: C[i,j] at j*2+i)
    # C11 -> W[0], C12 -> W[1], C21 -> W[2], C22 -> W[3]
    
    # M1 = (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[3][0] = 1  # A11, A22
    V2[0][0] = 1; V2[3][0] = 1  # B11, B22
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1  # All C entries get +1 from M1
    
    # M2 = (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1  # A21, A22
    V2[0][1] = 1  # B11
    W2[2][1] = 1; W2[3][1] = 1  # C21, C22 get +1 from M2
    
    # M3 = A11(B12-B22)
    U2[0][2] = 1  # A11
    V2[1][2] = 1; V2[3][2] = -1  # B12, -B22
    W2[1][2] = 1  # C12 gets +1 from M3
    
    # M4 = A22(B21-B11)
    U2[3][3] = 1  # A22
    V2[2][3] = 1; V2[0][3] = -1  # B21, -B11
    W2[0][3] = -1; W2[1][3] = 1  # C11 gets -1, C12 gets +1 from M4
    
    # M5 = (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1  # A11, A12
    V2[3][4] = 1  # B22
    W2[3][4] = -1  # C22 gets -1 from M5
    
    # M6 = (A21-A11)(B11+B12)
    U2[2][5] = 1; U2[0][5] = -1  # A21, -A11
    V2[0][5] = 1; V2[1][5] = 1  # B11, B12
    W2[0][5] = -1; W2[2][5] = 1  # C11 gets -1, C21 gets +1 from M6
    
    # M7 = (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1  # A12, -A22
    V2[2][6] = 1; V2[3][6] = 1  # B21, B22
    W2[0][6] = 1; W2[3][6] = 1  # C11 gets +1, C22 gets +1 from M7
    
    # Verify: 
    # C11 = M1 - M4 - M6 + M7: W2[0] = 1 - (-1) - (-1) + 1 = 1+1+1+1 = 4? That's wrong.
    # Let me re-check the Strassen formulas.
    
    # Actually, I think I'm confusing myself. Let me use a known correct Strassen decomposition.
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M4
    # C21 = M2 + M6
    # C22 = M1 - M2 - M5 + M7
    
    # Let me recompute W2 carefully.
    # For each Strassen step r, W2[i][r] is the coefficient of C[i] in the sum.
    
    # Reset and redo
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A11+A22)(B11+B22)
    # A: A11=1, A22=1 -> U2[0][0]=1, U2[3][0]=1
    # B: B11=1, B22=1 -> V2[0][0]=1, V2[3][0]=1
    # C: M1 contributes to C11, C12, C21, C22 with coeffs from formulas
    # C11: +1, C12: 0, C21: 0, C22: +1 (from M1 in C11 and C22)
    # Wait, M1 is a 2x2 matrix. Its (1,1) entry goes to C11, (1,2) to C12, etc.
    # But in the tensor decomposition, each "multiplication" M_r is itself a 2x2x2 tensor.
    # So M_r contributes to all 4 C entries.
    # The coefficient in W2 tells us how much of M_r's output goes to each C entry.
    
    # This is getting complicated. Let me use a different approach.
    
    # I'll directly construct the 4x4x4 Strassen decomposition by expanding.
    
    # For 4x4x4 with 2x2 blocks:
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3), C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M4
    # C21 = M2 + M6
    # C22 = M1 - M2 - M5 + M7
    
    # Each M_r is a 2x2x2 multiplication, which has rank 7.
    # So total rank for 4x4x4 is 7*7 = 49.
    
    # Let's build this.
    # U4 shape: (16, 49)
    # V4 shape: (16, 49)  
    # W4 shape: (16, 49)
    
    # A indices (row-major 4x4): A[i,k] at i*4+k
    # A11: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # A12: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # A21: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # A22: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    
    # B indices (row-major 4x4): B[k,j] at k*4+j
    # B11: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # B12: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # B21: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # B22: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    
    # C indices (W convention: C[i,j] at j*4+i)
    # C11: (0,0),(0,1),(1,0),(1,1) -> W[0],W[1],W[4],W[5]
    # C12: (0,2),(0,3),(1,2),(1,3) -> W[8],W[9],W[12],W[13]
    # C21: (2,0),(2,1),(3,0),(3,1) -> W[2],W[3],W[6],W[7]
    # C22: (2,2),(2,3),(3,2),(3,3) -> W[10],W[11],W[14],W[15]
    
    # For each Strassen step r (0-6), we have a 2x2x2 decomposition with rank 7.
    # The 2x2x2 U2, V2, W2 are defined above (but I need to fix them).
    
    # Let me use a known correct 2x2x2 Strassen.
    # I'll look up the standard one.
    
    # Actually, let me just use schoolbook for 2x2x2 (rank 4) to be safe, 
    # then total rank for 4x4x4 is 7*4 = 28.
    # And 4x4x5 is 28 + 16 = 44.
    # Total for 4x4x9: 28 + 28 + 44 = 100.
    
    # Hmm, that's not as good as I hoped. Let me try to get the Strassen right.
    
    # I'll construct the full 4x4x4x9 decomposition directly using a block Strassen approach.
    
    # Let me use a simpler strategy: 
    # 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # Use rank 19 for each 4x4x4 (if I can construct it) and rank 16 for 4x4x1.
    # Total: 19 + 19 + 16 = 54.
    
    # But I need to construct rank-19 4x4x4. Let me try.
    
    # Known: 4x4x4 rank is 19. One construction uses 2x2 Strassen with some optimizations.
    
    # Given the complexity, let me use a pragmatic approach:
    # Use 2x2 Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x1 (rank 16).
    # Total: 49 + 49 + 16 = 114.
    
    # Let me implement this correctly.
    
    # First, build the 2x2x2 Strassen decomposition properly.
    
    # 2x2x2: A (2x2), B (2x2), C = A*B (2x2)
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices (W convention): C[i,j] at j*2+i
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M2 - M5 + M7
    
    # Each M_r is a scalar product of linear forms in A and B.
    # M_r = (sum_i u_r[i] * A[i]) * (sum_j v_r[j] * B[j])
    # And C[i] = sum_r w_r[i] * M_r
    
    # So U2[i][r] = u_r[i], V2[j][r] = v_r[j], W2[i][r] = w_r[i]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1  # C00 and C11 get +1 from M1
    
    # r=1: M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1  # C10 gets +1, C11 gets -1 from M2
    
    # r=2: M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1  # C01 gets +1 from M3
    
    # r=3: M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = -1; W2[1][3] = 1  # C00 gets -1, C01 gets +1 from M4
    
    # r=4: M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[3][4] = -1  # C11 gets -1 from M5
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = -1; W2[2][5] = 1  # C00 gets -1, C10 gets +1 from M6
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1  # C00 gets +1, C11 gets +1 from M7
    
    # Verify C00 = M1 - M4 - M6 + M7:
    # W2[0] = 1 - (-1) - (-1) + 1 = 1+1+1+1 = 4. That's wrong, should be 1.
    
    # I think the issue is that M_r are 2x2 matrices, not scalars.
    # Each M_r = A_r * B_r where A_r and B_r are 2x2 matrices.
    # So M_r has 4 entries, and each entry contributes to a specific C entry.
    
    # This means the decomposition is more complex. Each "multiplication" M_r 
    # is itself a 2x2x2 tensor, not a scalar.
    
    # Given the time constraints, let me use a simpler but correct approach:
    # Use schoolbook for everything, but with a better block structure.
    
    # 4x4x9: schoolbook rank 144.
    # Let me try to improve by splitting into 4x4x4 + 4x4x4 + 4x4x1.
    # 4x4x4 schoolbook: rank 16.
    # 4x4x1 schoolbook: rank 16.
    # Total: 16 + 16 + 16 = 48.
    
    # Wait, that's better! Let me implement this.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 4x4x4 for columns 0-3
    for i in range(4):
        for k in range(4):
            for j in range(4):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # 4x4x4 for columns 4-7
    for i in range(4):
        for k in range(4):
            for j in range(4, 8):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # 4x4x1 for column 8
    for i in range(4):
        for k in range(4):
            U[i * m + k][col] = 1
            V[k * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
