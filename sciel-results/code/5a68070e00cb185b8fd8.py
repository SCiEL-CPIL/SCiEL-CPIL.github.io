import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7

    # Winograd-style decomposition using polynomial evaluation/interpolation.
    # We embed the 4x4x7 multiplication into a cyclic convolution structure.
    # Pad m from 4 to 8 (next power of 2) to use FFT-like decomposition.
    # The key idea: use the fact that matrix multiplication can be viewed as
    # a bilinear form, and use the Winograd algorithm with a suitable ring.
    
    # For simplicity and correctness, I'll implement a rank-97 decomposition:
    # - Strassen for the 4x4x4 part (rank 49)
    # - Schoolbook for the 4x4x3 part (rank 48)
    # Total: 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97

    # Part 1: Strassen 4x4x4 for the first 4 columns of B
    # Strassen 2x2x2 base case vectors (U, V, W are 4x7 each)
    # Standard Strassen 7 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4x4, we use 2x2 blocks of the 4x4 matrices.
    # The 4x4 matrix A is divided into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # Similarly for B and C.
    
    # Strassen 4x4x4 has rank 7*7 = 49 (applying Strassen to each 2x2 block multiplication)
    # We need to construct the 49 terms explicitly.
    
    # Let me define the Strassen 2x2x2 decomposition first.
    # U_strassen (4x7), V_strassen (4x7), W_strassen (4x7)
    # Row indices: (0,0), (0,1), (1,0), (1,1) -> flattened as 0,1,2,3
    # Col indices: 0..6 for the 7 terms
    
    U_s = [
        [1, 1, 1, 0, 1, 0, 0],  # A11
        [0, 0, 0, 1, 0, 1, -1], # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 0, 0, 1, 0, 0, 1],  # A22
    ]
    V_s = [
        [1, 1, 0, 1, 0, 1, 0],  # B11
        [0, 0, 1, -1, 1, 0, 1], # B12
        [0, 0, 1, 0, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 0], # B22
    ]
    W_s = [
        [1, 0, 0, 0, 0, 0, 0],  # C11 = M1 - M4 + M6 - M7
        [0, 0, 0, 0, 1, 0, 0],  # C12 = M5 + M7
        [0, 0, 0, 0, 0, 1, 0],  # C21 = M2 + M4
        [0, 1, 0, 0, 0, 0, 0],  # C22 = M1 - M3 + M5 - M6
    ]
    
    # Let me verify the Strassen 2x2x2 decomposition.
    # C11 = M1 - M4 + M6 - M7
    # M1 = (A11+A22)(B11+B22)
    # M4 = A22(B12-B11)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 should be A11*B11 + A12*B21
    
    # Let me re-derive. The standard Strassen formulas are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M4 + M6 - M7
    # C12 = M5 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # Let me check C11 = M1 - M4 + M6 - M7:
    # M1 - M4 + M6 - M7
    # = (A11+A22)(B11+B22) - A22(B12-B11) + (A21-A11)(B11+B12) - (A12-A22)(B21+B22)
    # = A11B11 + A11B22 + A22B11 + A22B22 - A22B12 + A22B11 + A21B11 + A21B12 - A11B11 - A11B12 - A12B21 - A12B22 + A22B21 + A22B22
    # = A11B11 + A11B22 + A22B11 + A22B22 - A22B12 + A22B11 + A21B11 + A21B12 - A11B11 - A11B12 - A12B21 - A12B22 + A22B21 + A22B22
    # Simplifying:
    # A11B11 - A11B11 + A11B22 - A11B12 + A22B11 + A22B11 + A22B22 + A22B22 - A22B12 + A22B21 + A21B11 + A21B12 - A12B21 - A12B22
    # = A11B22 - A11B12 + 2A22B11 + 2A22B22 - A22B12 + A22B21 + A21B11 + A21B12 - A12B21 - A12B22
    # This doesn't simplify to A11B11 + A12B21.
    
    # I think I have the W matrix wrong. Let me look up the correct Strassen 2x2x2 vectors.
    
    # The correct decomposition:
    # U (for A):
    # Term 0: A11+A22 -> [1, 0, 0, 1]
    # Term 1: A21+A22 -> [0, 0, 1, 1]
    # Term 2: A11 -> [1, 0, 0, 0]
    # Term 3: A22 -> [0, 0, 0, 1]
    # Term 4: A11+A12 -> [1, 1, 0, 0]
    # Term 5: A21-A11 -> [-1, 0, 1, 0]
    # Term 6: A12-A22 -> [0, 1, 0, -1]
    
    # V (for B):
    # Term 0: B11+B22 -> [1, 0, 0, 1]
    # Term 1: B11 -> [1, 0, 0, 0]
    # Term 2: B21-B22 -> [0, 0, 1, -1]
    # Term 3: B12-B11 -> [-1, 1, 0, 0]
    # Term 4: B22 -> [0, 0, 0, 1]
    # Term 5: B11+B12 -> [1, 1, 0, 0]
    # Term 6: B21+B22 -> [0, 0, 1, 1]
    
    # W (for C):
    # C11 = M0 - M3 + M5 - M6 -> [1, 0, 0, -1, 0, 1, -1]
    # C12 = M4 + M6 -> [0, 0, 0, 0, 1, 0, 1]
    # C21 = M1 + M3 -> [0, 1, 0, 1, 0, 0, 0]
    # C22 = M0 - M2 + M4 - M5 -> [1, 0, -1, 0, 1, -1, 0]
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, 1],    # A12
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [1, 1, 0, 1, 0, 0, -1],   # A22
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 0, 1, 0, 1, 0],    # B12
        [0, 0, 1, 0, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1],   # B22
    ]
    W_s = [
        [1, 0, 0, -1, 0, 1, -1],  # C11
        [0, 0, 0, 0, 1, 0, 1],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, 0, -1, 0, 1, -1, 0],  # C22
    ]
    
    # Verify: sum over t of U_s[i,t] * V_s[j,t] * W_s[k,t] should equal delta(i1,j1)*delta(i2,j2) for 2x2
    # For i=0 (A11), j=0 (B11), k=0 (C11):
    # sum_t U_s[0,t]*V_s[0,t]*W_s[0,t]
    # = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*(-1) + 1*0*0 + (-1)*1*1 + 0*0*(-1)
    # = 1 + 0 + 0 + 0 + 0 + (-1) + 0 = 0. But should be 1.
    
    # Something is wrong. Let me be more careful.
    
    # Let me just use the schoolbook for everything to ensure correctness, but with a rank that's an improvement.
    # Actually, the simplest correct improvement is to use the schoolbook but recognize that we can't reduce below n*m*p for a general decomposition.
    
    # Wait, the schoolbook IS the decomposition with rank n*m*p = 112.
    # To get an improvement, I need a genuine algorithmic improvement.
    
    # Let me try a different approach: use the fact that 4x4x7 can be decomposed using
    # the Winograd algorithm with a specific structure.
    
    # For now, let me just return the schoolbook to ensure validity.
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
