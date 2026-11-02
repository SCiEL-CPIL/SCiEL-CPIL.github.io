import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 blocks of (2x2x2 rank 7 + 2x1x2 rank 4) = 3 * 11 = 33
    # 1 block of (2x3x1 rank 6) = 6
    # Total rank = 39
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_222(offset_u, offset_v, offset_w, col_start):
        # A is 2x2 at offset_u, B is 2x2 at offset_v, C is 2x2 at offset_w
        # Strassen M1..M7
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices for A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Indices for B: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7 (relative to m*p start? No, relative to V start)
        # V is m*p. m=3, p=7.
        # For a 2x2 block of B, we need to know which 2x2 subblock.
        
        # This helper is too complex for generic offsets without knowing the exact subblock positions.
        # I will hardcode the indices for each block.
        pass

    # Block 1: p indices 0,1
    # A is full 2x3.
    # A_left (cols 0,1) is 2x2. A_right (col 2) is 2x1.
    # B1 (rows 0,1,2; cols 0,1) is 3x2.
    # B1_top (rows 0,1) is 2x2. B1_bot (row 2) is 1x2.
    
    # 1.1: A_left * B1_top (2x2x2)
    # U indices for A_left: 0,1,2,3
    # V indices for B1_top: 0,1,4,5 (j=0,1; k=0,1) -> j*p+k. 0*7+0=0, 0*7+1=1, 1*7+0=7, 1*7+1=8.
    # Wait, V index is j*p + k.
    # B1_top is rows 0,1 of B. Cols 0,1.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 7
    # j=1, k=1 -> 8
    # W indices for C1: k*n+i. k=0,1. i=0,1.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # M1 = (a+d)(e+h)
    U[0, col] = 1; U[3, col] = 1
    V[0, col] = 1; V[8, col] = 1
    W[0, col] = 1; W[1, col] = 1; W[2, col] = 1; W[3, col] = 1
    col += 1
    
    # M2 = (c+d)e
    U[2, col] = 1; U[3, col] = 1
    V[0, col] = 1
    W[2, col] = 1; W[3, col] = 1
    col += 1
    
    # M3 = a(f-h)
    U[0, col] = 1
    V[1, col] = 1; V[8, col] = -1
    W[1, col] = 1; W[3, col] = 1
    col += 1
    
    # M4 = d(g-e)
    U[3, col] = 1
    V[7, col] = 1; V[0, col] = -1
    W[1, col] = -1; W[3, col] = 1
    col += 1
    
    # M5 = (a+b)h
    U[0, col] = 1; U[1, col] = 1
    V[8, col] = 1
    W[1, col] = 1; W[3, col] = 1
    col += 1
    
    # M6 = (c-a)(e+f)
    U[2, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[0, col] = 1; W[1, col] = -1; W[2, col] = 1
    col += 1
    
    # M7 = (b-d)(g+h)
    U[1, col] = 1; U[3, col] = -1
    V[7, col] = 1; V[8, col] = 1
    W[0, col] = -1; W[1, col] = -1; W[2, col] = 1
    col += 1
    
    # 1.2: A_right * B1_bot (2x1x2)
    # A_right is col 2 of A. Indices 4, 5.
    # B1_bot is row 2 of B1. Cols 0,1.
    # j=2, k=0 -> 14
    # j=2, k=1 -> 15
    # C1 indices same as above.
    
    # M1 = A[0,2] * B[2,0]
    U[4, col] = 1
    V[14, col] = 1
    W[0, col] = 1
    col += 1
    
    # M2 = A[0,2] * B[2,1]
    U[4, col] = 1
    V[15, col] = 1
    W[1, col] = 1
    col += 1
    
    # M3 = A[1,2] * B[2,0]
    U[5, col] = 1
    V[14, col] = 1
    W[2, col] = 1
    col += 1
    
    # M4 = A[1,2] * B[2,1]
    U[5, col] = 1
    V[15, col] = 1
    W[3, col] = 1
    col += 1

    # Block 2: p indices 2,3
    # B2_top: j=0,1; k=2,3.
    # V indices: 0*7+2=2, 0*7+3=3, 1*7+2=9, 1*7+3=10
    # B2_bot: j=2; k=2,3.
    # V indices: 2*7+2=16, 2*7+3=17
    # C2: k=2,3. i=0,1.
    # W indices: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7
    
    # 2.1: A_left * B2_top
    U[0, col] = 1; U[3, col] = 1
    V[2, col] = 1; V[10, col] = 1
    W[4, col] = 1; W[5, col] = 1; W[6, col] = 1; W[7, col] = 1
    col += 1
    
    U[2, col] = 1; U[3, col] = 1
    V[2, col] = 1
    W[6, col] = 1; W[7, col] = 1
    col += 1
    
    U[0, col] = 1
    V[3, col] = 1; V[10, col] = -1
    W[5, col] = 1; W[7, col] = 1
    col += 1
    
    U[3, col] = 1
    V[9, col] = 1; V[2, col] = -1
    W[5, col] = -1; W[7, col] = 1
    col += 1
    
    U[0, col] = 1; U[1, col] = 1
    V[10, col] = 1
    W[5, col] = 1; W[7, col] = 1
    col += 1
    
    U[2, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[4, col] = 1; W[5, col] = -1; W[6, col] = 1
    col += 1
    
    U[1, col] = 1; U[3, col] = -1
    V[9, col] = 1; V[10, col] = 1
    W[4, col] = -1; W[5, col] = -1; W[6, col] = 1
    col += 1
    
    # 2.2: A_right * B2_bot
    U[4, col] = 1
    V[16, col] = 1
    W[4, col] = 1
    col += 1
    
    U[4, col] = 1
    V[17, col] = 1
    W[5, col] = 1
    col += 1
    
    U[5, col] = 1
    V[16, col] = 1
    W[6, col] = 1
    col += 1
    
    U[5, col] = 1
    V[17, col] = 1
    W[7, col] = 1
    col += 1

    # Block 3: p indices 4,5
    # B3_top: j=0,1; k=4,5.
    # V indices: 4, 5, 11, 12
    # B3_bot: j=2; k=4,5.
    # V indices: 18, 19
    # C3: k=4,5. i=0,1.
    # W indices: 8, 9, 10, 11
    
    # 3.1: A_left * B3_top
    U[0, col] = 1; U[3, col] = 1
    V[4, col] = 1; V[12, col] = 1
    W[8, col] = 1; W[9, col] = 1; W[10, col] = 1; W[11, col] = 1
    col += 1
    
    U[2, col] = 1; U[3, col] = 1
    V[4, col] = 1
    W[10, col] = 1; W[11, col] = 1
    col += 1
    
    U[0, col] = 1
    V[5, col] = 1; V[12, col] = -1
    W[9, col] = 1; W[11, col] = 1
    col += 1
    
    U[3, col] = 1
    V[11, col] = 1; V[4, col] = -1
    W[9, col] = -1; W[11, col] = 1
    col += 1
    
    U[0, col] = 1; U[1, col] = 1
    V[12, col] = 1
    W[9, col] = 1; W[11, col] = 1
    col += 1
    
    U[2, col] = 1; U[0, col] = -1
    V[4, col] = 1; V[5, col] = 1
    W[8, col] = 1; W[9, col] = -1; W[10, col] = 1
    col += 1
    
    U[1, col] = 1; U[3, col] = -1
    V[11, col] = 1; V[12, col] = 1
    W[8, col] = -1; W[9, col] = -1; W[10, col] = 1
    col += 1
    
    # 3.2: A_right * B3_bot
    U[4, col] = 1
    V[18, col] = 1
    W[8, col] = 1
    col += 1
    
    U[4, col] = 1
    V[19, col] = 1
    W[9, col] = 1
    col += 1
    
    U[5, col] = 1
    V[18, col] = 1
    W[10, col] = 1
    col += 1
    
    U[5, col] = 1
    V[19, col] = 1
    W[11, col] = 1
    col += 1

    # Block 4: p index 6
    # B4: j=0,1,2; k=6.
    # V indices: 6, 13, 20
    # C4: k=6. i=0,1.
    # W indices: 12, 13
    
    # 4.1: A * B4 (2x3x1)
    # A[0,0] B[0,6]
    U[0, col] = 1
    V[6, col] = 1
    W[12, col] = 1
    col += 1
    
    # A[0,1] B[1,6]
    U[1, col] = 1
    V[13, col] = 1
    W[12, col] = 1
    col += 1
    
    # A[0,2] B[2,6]
    U[2, col] = 1
    V[20, col] = 1
    W[12, col] = 1
    col += 1
    
    # A[1,0] B[0,6]
    U[3, col] = 1
    V[6, col] = 1
    W[13, col] = 1
    col += 1
    
    # A[1,1] B[1,6]
    U[4, col] = 1
    V[13, col] = 1
    W[13, col] = 1
    col += 1
    
    # A[1,2] B[2,6]
    U[5, col] = 1
    V[20, col] = 1
    W[13, col] = 1
    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
