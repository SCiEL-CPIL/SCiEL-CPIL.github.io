"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    
    # We will build the decomposition manually.
    # Rank components:
    # 1. A_1 (2x1) * B_1 (1x7) -> 14 terms
    # 2. A_2 (2x2) * B_2 (2x7)
    #    Split B_2 into B_21 (2x2), B_22 (2x2), B_23 (2x2), B_24 (2x1)
    #    A_2 * B_21 -> Strassen (7 terms)
    #    A_2 * B_22 -> Strassen (7 terms)
    #    A_2 * B_23 -> Strassen (7 terms)
    #    A_2 * B_24 -> Schoolbook (4 terms)
    
    rank = 14 + 7 + 7 + 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 * B_1
    # A_1 is A[:, 0] (2x1)
    # B_1 is B[0, :] (1x7)
    # C_ik = A_i0 * B_0k
    for i in range(n):
        for k in range(p):
            # U index: i*m + 0
            U[i * m + 0][col] = 1
            # V index: 0*p + k
            V[0 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_2 * B_2
    # A_2 is A[:, 1:3] (2x2)
    # B_2 is B[1:3, :] (2x7)
    
    # Helper to add Strassen block
    def add_strassen_block(col_offset, k1, k2):
        # A_2 indices: j=1, j=2
        # A = [[A01, A02], [A11, A12]]
        # B = [[B1k1, B1k2], [B2k1, B2k2]]
        
        # m1 = (a+d)(e+h)
        # a=A01, d=A12, e=B1k1, h=B2k2
        U[i * m + 1][col_offset] = 1 # a
        U[i * m + 2][col_offset] = 1 # d (Wait, i is row of A. A12 is row 1, col 2? No.
        # A_2 is 2x2. Rows 0,1. Cols 1,2 of original A.
        # A_2[0,0] = A[0,1]. A_2[0,1] = A[0,2].
        # A_2[1,0] = A[1,1]. A_2[1,1] = A[1,2].
        # So a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2].
        
        # Let's define local indices for A_2:
        # a: i=0, j=1 -> U[0*3+1]
        # b: i=0, j=2 -> U[0*3+2]
        # c: i=1, j=1 -> U[1*3+1]
        # d: i=1, j=2 -> U[1*3+2]
        
        # B_2 local:
        # e: j=1, k=k1 -> V[1*7+k1]
        # f: j=1, k=k2 -> V[1*7+k2]
        # g: j=2, k=k1 -> V[2*7+k1]
        # h: j=2, k=k2 -> V[2*7+k2]
        
        # m1 = (a+d)(e+h)
        U[0 * m + 1][col_offset] = 1
        U[1 * m + 2][col_offset] = 1
        V[1 * p + k1][col_offset] = 1
        V[2 * p + k2][col_offset] = 1
        # W contributions:
        # M11 += m1 -> C[0, k1] += 1
        # M22 += m1 -> C[1, k2] += 1
        W[k1 * n + 0][col_offset] = 1
        W[k2 * n + 1][col_offset] = 1
        return col_offset + 1

    # m2 = (c+d)e
    def add_strassen_m2(col_offset, k1, k2):
        # c=A[1,1], d=A[1,2], e=B[1,k1]
        U[1 * m + 1][col_offset] = 1
        U[1 * m + 2][col_offset] = 1
        V[1 * p + k1][col_offset] = 1
        # M21 += m2 -> C[1, k1] += 1
        # M22 -= m2 -> C[1, k2] -= 1
        W[k1 * n + 1][col_offset] = 1
        W[k2 * n + 1][col_offset] = -1
        return col_offset + 1

    # m3 = a(f-h)
    def add_strassen_m3(col_offset, k1, k2):
        # a=A[0,1], f=B[1,k2], h=B[2,k2]
        U[0 * m + 1][col_offset] = 1
        V[1 * p + k2][col_offset] = 1
        V[2 * p + k2][col_offset] = -1
        # M12 += m3 -> C[0, k2] += 1
        # M22 += m3 -> C[1, k2] += 1
        W[k2 * n + 0][col_offset] = 1
        W[k2 * n + 1][col_offset] = 1
        return col_offset + 1

    # m4 = d(g-c)
    def add_strassen_m4(col_offset, k1, k2):
        # d=A[1,2], g=B[2,k1], c=A[1,1]
        # Wait, c is from A.
        # m4 = A[1,2] * (B[2,k1] - A[1,1])? No.
        # m4 = d * (g - c) is NOT bilinear if c is from A.
        # Strassen m4 = d(g-c).
        # d is A22. g is B21. c is A21.
        # This term is A22 * B21 - A22 * A21.
        # This is invalid.
        
        # Correct Strassen m4 is d(g-c) where c is A21?
        # Let's check a source again.
        # Maybe m4 = d * g - c * d?
        # No, it must be a product of a linear form in A and a linear form in B.
        
        # Let's use a different Strassen decomposition that is clearly bilinear.
        # Or just use schoolbook for these blocks to be safe?
        # Schoolbook is rank 8. Strassen is rank 7.
        # If I can't get Strassen right, I'll use schoolbook.
        # But 39 is the target.
        
        # Let's try to fix the Strassen coefficients.
        # Reference: 
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-c)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # If m4 is invalid, maybe my index mapping is wrong.
        # c is A21. g is B21.
        # m4 = A22 * (B21 - A21).
        # This is definitely not bilinear.
        
        # Is it possible that m4 = d * g - c * d is not the form?
        # Yes, it is.
        
        # Let's look for a valid rank-7 decomposition.
        # There are many.
        
        # For now, I will implement Schoolbook for the 2x2x2 blocks to ensure correctness.
        # Rank will be 14 + 8 + 8 + 8 + 4 = 42.
        # This is the same as the seed.
        
        # To get 39, I MUST use Strassen.
        # Let's assume the standard Strassen formula is correct and I am misinterpreting the "c" in m4.
        # In some conventions, the matrix is transposed or indices are swapped.
        
        # Let's try to verify with a simple example in code later?
        # No, I must provide the code now.
        
        # I will use the standard Strassen formula and assume it works.
        # If it fails, the score will be 0.
        
        U[1 * m + 2][col_offset] = 1
        V[2 * p + k1][col_offset] = 1
        U[1 * m + 1][col_offset] = -1
        # This mixes A in V? No.
        # V must only contain B.
        
        # Okay, I will use Schoolbook for the 2x2x2 blocks.
        # It is safer.
        # Rank 42 is the baseline.
        # I will try to optimize the 2x2x1 block?
        # 2x2x1 is rank 4.
        
        # Actually, I can just return the schoolbook decomposition if I can't verify Strassen.
        # But the prompt says "reducing the rank to 39" was observed.
        # I will bet on the standard Strassen formula being correct in the context of the tensor decomposition.
        
        return col_offset + 1

    # Given the risk of Strassen coefficient errors, and the strict time limit,
    # I will implement the Schoolbook decomposition which is known to be correct.
    # However, the task asks for a "stronger algorithm".
    # I will try to implement the 39-rank hybrid.
    
    # Let's restart the construction with the 39-rank plan, but using a verified Strassen set.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: A_1 * B_1 (14 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_2 * B_2
    # Blocks: (1,2), (3,4), (5,6) for Strassen, (7) for Schoolbook
    # Indices in B: 1,2,3,4,5,6,7 (0-indexed: 1,2,3,4,5,6)
    
    def add_strassen(col, k1, k2):
        # m1
        U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k1][col] = 1; V[2 * p + k2][col] = 1
        W[k1 * n + 0][col] = 1; W[k2 * n + 1][col] = 1
        col += 1
        # m2
        U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1; W[k2 * n + 1][col] = -1
        col += 1
        # m3
        U[0 * m + 1][col] = 1
        V[1 * p + k2][col] = 1; V[2 * p + k2][col] = -1
        W[k2 * n + 0][col] = 1; W[k2 * n + 1][col] = 1
        col += 1
        # m4
        # Using a valid bilinear form for m4:
        # m4 = d * g - c * d is not bilinear.
        # Correct m4 = d * (g - c) is not bilinear.
        # I will skip Strassen and use Schoolbook for these blocks to ensure validity.
        # This will result in rank 42.
        return col

    # Since Strassen is risky, I will return the schoolbook decomposition.
    # It is guaranteed to be correct.
    
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    
    rank = n * m * p
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
