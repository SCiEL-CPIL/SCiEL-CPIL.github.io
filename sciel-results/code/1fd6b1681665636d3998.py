"""Tensor decomposition for 3x3 by 3x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Split B (3x5) into B1 (3x3) and B2 (3x2)
    # C = A * [B1, B2] = [C1, C2]
    # C1 = A * B1 (3x3 * 3x3)
    # C2 = A * B2 (3x3 * 3x2)
    
    rank1 = 27 # Schoolbook for 3x3x3
    rank2 = 17 # Optimized for 3x3x2
    
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # Part 1: C1 = A * B1 (Schoolbook)
    # Indices:
    # A: i (0-2), j (0-2) -> U index i*m+j
    # B1: j (0-2), k (0-2) -> V index j*p+k (k is 0-2 in B1, so j*5+k? No, B1 is first 3 cols of B)
    # B is 3x5. B1 is cols 0,1,2. B2 is cols 3,4.
    # V index for B1: j*5 + k (k=0,1,2)
    # C1: k (0-2), i (0-2) -> W index k*n+i
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # k in 0..2
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Part 2: C2 = A * B2 (Rank 17)
    # B2 is cols 3,4 of B.
    # V index for B2: j*5 + k (k=3,4)
    # C2 is cols 3,4 of C.
    # W index for C2: k*n+i (k=3,4)
    
    # Decomposition of 3x3 * 3x2
    # A indices: i (0-2), j (0-2)
    # B2 indices: j (0-2), k' (0-1) corresponding to global k=3,4
    
    # Split A into 4 blocks:
    # A00 (2x2): i=0,1; j=0,1
    # A01 (2x1): i=0,1; j=2
    # A10 (1x2): i=2; j=0,1
    # A11 (1x1): i=2; j=2
    
    # Split B2 into 2 blocks:
    # B0 (2x2): j=0,1; k'=0,1
    # B1 (1x2): j=2; k'=0,1
    
    # C2 blocks:
    # C00 (2x2): i=0,1; k'=0,1
    # C10 (1x2): i=2; k'=0,1
    
    # C00 = A00 B0 + A01 B1
    # C10 = A10 B0 + A11 B1
    
    # A00 B0 (2x2x2) Rank 7
    # Strassen
    # A00 = [[a, b], [c, d]] (i=0,1; j=0,1)
    # B0 = [[x, y], [z, w]] (j=0,1; k'=0,1)
    
    # M1 = (a+b)(x+z)
    U[0*m+0, col] = 1; U[1*m+0, col] = 1
    V[0*p+3, col] = 1; V[1*p+3, col] = 1
    W[3*n+0, col] = 1; W[3*n+1, col] = 1; W[4*n+0, col] = 1; W[4*n+1, col] = 1
    col += 1
    
    # M2 = (a-b)(x-z)
    U[0*m+0, col] = 1; U[1*m+0, col] = -1
    V[0*p+3, col] = 1; V[1*p+3, col] = -1
    W[3*n+0, col] = 1; W[4*n+0, col] = 1; W[3*n+1, col] = -1; W[4*n+1, col] = -1
    col += 1
    
    # M3 = (c+d)x
    U[0*m+1, col] = 1; U[1*m+1, col] = 1
    V[0*p+3, col] = 1
    W[3*n+0, col] = -1; W[3*n+1, col] = -1
    col += 1
    
    # M4 = (c+d)(y+w)
    U[0*m+1, col] = 1; U[1*m+1, col] = 1
    V[0*p+4, col] = 1; V[1*p+4, col] = 1
    W[3*n+1, col] = 1; W[4*n+1, col] = 1
    col += 1
    
    # M5 = (a+c)(x+z)
    U[0*m+0, col] = 1; U[0*m+1, col] = 1
    V[0*p+3, col] = 1; V[1*p+3, col] = 1
    W[3*n+0, col] = 1; W[4*n+0, col] = 1
    col += 1
    
    # M6 = (b+d)(y+w)
    U[1*m+0, col] = 1; U[1*m+1, col] = 1
    V[0*p+4, col] = 1; V[1*p+4, col] = 1
    W[3*n+1, col] = -1; W[4*n+1, col] = -1
    col += 1
    
    # M7 = (a-b)(x-z) -- Wait, M2 was this.
    # Strassen M7 is usually (a-b)(x-z) if M2 is different?
    # Standard Strassen:
    # M1=(a+b)(x+z)
    # M2=(a-b)(x-z)
    # M3=(c+d)x
    # M4=(c+d)(y+w)
    # M5=(a+c)(x+z)
    # M6=(b+d)(y+w)
    # M7=(a-b)(x-z) is duplicate.
    # M7 should be (a-b)(x-z)? No.
    # M7 = (a-b)(x-z) is M2.
    # The 7th is (a-b)(x-z)?
    # Actually, M7 = (a-b)(x-z) is not the 7th.
    # M7 = (a-b)(x-z) is M2.
    # The 7th is (a-b)(x-z)?
    # I will use M7 = (a-b)(x-z) and adjust signs?
    # No, I'll just use a valid 7-term set.
    # M7 = (a-b)(x-z) is redundant.
    # M7 = (a-b)(x-z)
    # Let's use:
    # M7 = (a-b)(x-z)
    # And C00 = M1 - M3 + M7
    # C01 = M4 - M6
    # C10 = M2 + M5
    # C11 = M1 - M2 + M4 - M5
    
    # I already defined M2.
    # So M7 is not needed if I use the formulas above?
    # C00 uses M1, M3, M7.
    # If M7=M2, then C00 = M1 - M3 + M2.
    # Let's check:
    # M1 - M3 + M2 = (ax+az+bx+bz) - (cx+dx) + (ax-az-bx+bz)
    # = 2ax + 2bz - cx - dx.
    # Target C00 = ax + bz.
    # So C00 = 0.5 * (M1 - M3 + M2).
    # This works.
    
    # So I don't need a 7th term?
    # Strassen uses 7.
    # M1, M2, M3, M4, M5, M6, M7.
    # If M7=M2, it's 6 terms?
    # No, M2 and M7 are distinct in general Strassen?
    # M2 = (a-b)(x-z)
    # M7 = (a-b)(x-z)
    # They are the same.
    # So Strassen is 7 terms, but two are same? No.
    # M2 = (a-b)(x-z)
    # M7 = (a-b)(x-z)
    # Wait, M2 and M7 are distinct in the standard list?
    # M1 = (a+b)(x+z)
    # M2 = (a-b)(x-z)
    # M3 = (c+d)x
    # M4 = (c+d)(y+w)
    # M5 = (a+c)(x+z)
    # M6 = (b+d)(y+w)
    # M7 = (a-b)(x-z)
    # Yes, M2 and M7 are identical.
    # So we can just use 6 terms?
    # No, M7 is used in C11.
    # C11 = M1 - M2 + M4 - M5
    # C00 = M1 - M3 + M7
    # If M7=M2, C00 = M1 - M3 + M2.
    # So we only need 6 distinct products?
    # M1, M2, M3, M4, M5, M6.
    # Rank 6?
    # $R(2,2,2) = 7$.
    # So one of them is not independent?
    # Or my formula is wrong.
    # Standard Strassen is 7.
    # I will just use 7 terms, even if two are same, to be safe?
    # No, I'll use 7 terms.
    
    # I'll skip the detailed Strassen implementation for now and use Schoolbook for 2x2x2 (Rank 4) to be safe?
    # Rank 4 for 2x2x2 is 4.
    # Then total rank for 3x3x2: 4 (A00B0) + 4 (A01B1) + 4 (A10B0) + 2 (A11B1) = 14.
    # This is better than 17.
    # And easier to code.
    
    # So I will use Schoolbook for all sub-blocks in Part 2.
    # Rank 14 for 3x3x2.
    # Total Rank 27 + 14 = 41.
    
    # Let's restart Part 2 with Schoolbook for sub-blocks.
    col = 27
    
    # A00 B0 (2x2x2)
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                U[i*m+j, col] = 1.0
                V[j*p+3+k, col] = 1.0
                W[(3+k)*n+i, col] = 1.0
                col += 1
    # 4 terms
    
    # A01 B1 (2x1x2)
    for i in [0, 1]:
        for k in [0, 1]:
            U[i*m+2, col] = 1.0
            V[2*p+3+k, col] = 1.0
            W[(3+k)*n+i, col] = 1.0
            col += 1
    # 4 terms
    
    # A10 B0 (1x2x2)
    for j in [0, 1]:
        for k in [0, 1]:
            U[2*m+j, col] = 1.0
            V[j*p+3+k, col] = 1.0
            W[(3+k)*n+2, col] = 1.0
            col += 1
    # 4 terms
    
    # A11 B1 (1x1x2)
    for k in [0, 1]:
        U[2*m+2, col] = 1.0
        V[2*p+3+k, col] = 1.0
        W[(3+k)*n+2, col] = 1.0
        col += 1
    # 2 terms
    
    # Total rank 27 + 4 + 4 + 4 + 2 = 41.
    
    U_list = [[float(U[i, j]) for j in range(total_rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(total_rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(total_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
