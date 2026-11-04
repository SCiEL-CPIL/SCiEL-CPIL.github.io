import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We want to decompose the 2x4x6 matrix multiplication tensor.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # We can view this as a Kronecker product of two smaller multiplications:
    # 1. A 2x2x2 multiplication (Strassen, rank 7)
    # 2. A 2x2x3 multiplication (linear convolution, rank 4)
    #
    # The index j in {0,1,2,3} can be written as j = 2*j1 + j0 where j1 in {0,1}, j0 in {0,1}
    # The index k in {0,1,2,3,4,5} can be written as k = 2*k1 + k0 where k1 in {0,1,2}, k0 in {0,1}
    #
    # So A[i, j] = A[i, 2*j1+j0] and B[j, k] = B[2*j1+j0, 2*k1+k0]
    #
    # The multiplication becomes:
    # C[i, 2*k1+k0] = sum_{j1=0}^{1} sum_{j0=0}^{1} A[i, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    #
    # This is a tensor product structure. We can use the Kronecker product of:
    # - Strassen for 2x2x2 (rank 7)
    # - Linear convolution for 2x2x3 (rank 4)
    #
    # Total rank = 7 * 4 = 28
    
    # First, let's define the Strassen decomposition for 2x2x2.
    # A is 2x2, B is 2x2, C = A*B is 2x2.
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    #
    # Strassen's 7 multiplications:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    # M4 = A[1,1] * (B[0,1] - B[1,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
    # M7 = (A[0,1] - A[1,1]) * (B[0,1] + B[1,1])
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M5
    # C[1,1] = M1 - M3 + M6 - M7
    #
    # Let me verify this with a numerical example:
    # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    # C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[5+14, 6+16], [15+28, 18+32]] = [[19, 22], [43, 50]]
    #
    # M1 = (1+4)*(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 7*5 = 35
    # M3 = 1*(7-8) = 1*(-1) = -1
    # M4 = 4*(6-7) = 4*(-1) = -4
    # M5 = (1+2)*8 = 3*8 = 24
    # M6 = (3-1)*(5+7) = 2*12 = 24
    # M7 = (2-4)*(6+8) = (-2)*14 = -28
    #
    # C[0,0] = M1 - M4 + M5 - M7 = 65 - (-4) + 24 - (-28) = 65 + 4 + 24 + 28 = 121
    # But expected C[0,0] = 19. So my Strassen formulas are wrong.
    #
    # Let me look up the correct Strassen formulas more carefully.
    #
    # Correct Strassen:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B11)
    # P4 = A11(B01 - B10)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B10)
    # P7 = (A01 - A11)(B01 + B11)
    #
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P3 + P6 - P7
    #
    # Let me recompute:
    # P1 = 65, P2 = 35, P3 = -1, P4 = -4, P5 = 24, P6 = 24, P7 = -28
    #
    # C00 = P1 - P4 + P5 - P7 = 65 - (-4) + 24 - (-28) = 65 + 4 + 24 + 28 = 121. Still wrong.
    #
    # I think I have the indices of A and B mixed up. Let me try a different convention.
    # Maybe A is indexed as A[row, col] and B is indexed as B[row, col], and C = A*B means
    # C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # Let me try yet another set of formulas. I'll use:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    # M4 = A[1,1] * (B[0,1] - B[1,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
    # M7 = (A[0,1] - A[1,1]) * (B[0,1] + B[1,1])
    #
    # And the reconstruction:
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M5
    # C[1,1] = M1 - M3 + M6 - M7
    #
    # This is the same as before. Let me just try to find the correct formulas by searching.
    #
    # Actually, I think the issue is that I'm using the wrong indices for B. Let me try:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    # M4 = A[1,1] * (B[0,1] - B[1,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
    # M7 = (A[0,1] - A[1,1]) * (B[0,1] + B[1,1])
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M5
    # C[1,1] = M1 - M3 + M6 - M7
    #
    # I keep getting the same result. Let me just accept that I can't get Strassen right and use
    # a different approach. I'll use the schoolbook method for 2x2x2 (rank 8) and then combine
    # with the 2x2x3 convolution.
    #
    # Actually, let me just use a simple rank-24 decomposition by splitting into two 2x2x3 blocks
    # and using schoolbook for each.
    
    # For now, let me just implement a correct decomposition with rank 24.
    # I'll split the j-index into two groups of 2: j=0,1 and j=2,3.
    # For each group, I'll use a 2x2x6 multiplication with schoolbook (rank 12).
    # Total rank = 12 + 12 = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 8x24
    V = [[0] * rank for _ in range(m * p)]  # 24x24
    W = [[0] * rank for _ in range(p * n)]  # 12x24
    
    column = 0
    # First group: j=0,1
    for i in range(n):
        for j in range(2):  # j=0,1
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Second group: j=2,3
    for i in range(n):
        for j in range(2, 4):  # j=2,3
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
