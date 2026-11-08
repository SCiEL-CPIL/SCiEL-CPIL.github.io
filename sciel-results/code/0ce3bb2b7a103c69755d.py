import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Base decomposition for 2x2 x 2x3
    # Rank 11: 7 (Strassen for first 2 cols) + 4 (schoolbook for 3rd col)
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)
    W2 = np.zeros((6, 11), dtype=float)

    # Strassen part (cols 0-6)
    # p1 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1

    # p2 = (c+d)(e+h) ... wait, standard Strassen p2 is (c+d)e.
    # Let's use the standard 7 products for 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f) -> This is not standard.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = a(f-g)
    # M4 = d(h-e)
    # M5 = (a+b)(h+g)
    # M6 = (c+a)(h+f)
    # M7 = (b+d)(g-e)
    # But we are multiplying 2x2 A by 2x3 B.
    # Let A = [[a, b], [c, d]]
    # Let B = [[e, f, g], [h, i, j]]
    # C = [[ae+bg, af+bh, ag+bj], [ce+dg, cf+dh, cg+dj]]
    #
    # Using Strassen-like for the first two columns of B (e,f,h,i):
    # The "g" in standard Strassen corresponds to h.
    # The "h" in standard Strassen corresponds to i.
    #
    # M1 = (a+d)(e+h) -> contributes to C00, C01, C10, C11
    # M2 = (c+d)(e+f) -> No, M2 = (c+d)*e? No.
    # Let's just stick to the valid construction from the previous attempt which was score 0 (invalid) but the math was mostly right, or use a known valid 2x2x2x3.
    # Actually, the simplest valid approach that is "better" than schoolbook (96) is block decomposition.
    # If we use 2x2x2x3 of rank 11, and we have 8 blocks, total rank = 88.
    # Schoolbook for 4x4x6 is 96.
    # 88 < 96.
    #
    # Let's verify the 2x2x2x3 rank 11.
    # We can view it as 2x2x2 (rank 7) + 2x1x2 (rank 2) + 2x1x2 (rank 2)? No.
    # B is 2x3.
    # C[:,0:2] = A * B[:,0:2] (Rank 7 via Strassen)
    # C[:,2] = A * B[:,2] (Rank 4 via Schoolbook)
    # Total rank = 7 + 4 = 11.
    # This is a valid decomposition.

    # --- Constructing the 2x2x2x3 base matrices ---
    
    # Indices for A (2x2): 0=a, 1=b, 2=c, 3=d
    # Indices for B (2x3): 0=e, 1=f, 2=g, 3=h, 4=i, 5=j
    # Indices for C (2x3): 0=C00, 1=C10, 2=C01, 3=C11, 4=C02, 5=C12
    
    # Part 1: Strassen for B[:,0:2] = [[e,f],[h,i]]
    # Standard Strassen for 2x2:
    # A = [[a, b], [c, d]]
    # B' = [[e, f], [h, i]]
    # 
    # M1 = (a+d)(e+h)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[1,0]=1; W2[2,0]=1; W2[3,0]=1

    # M2 = (c+d)e  -- Note: Standard Strassen M2 is (c+d)(e+f) is WRONG.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)  <- NO.
    # Let's use the explicit coefficients from the seed program's U1/V1/W1 which were verified for 2x2x2.
    # The seed program had:
    # p1 = (a+d)(e+g)  [Here g is 2nd row 1st col of B in 2x2]
    # p2 = (c+d)e
    # p3 = a(f-g)
    # p4 = d(h-e)
    # p5 = (a+b)g
    # p6 = (c+a)(h+f)
    # p7 = (b+d)(g-e)
    
    # In our 2x3 case, B' = [[e,f],[h,i]].
    # So "g" in 2x2 Strassen maps to "h" (index 3) in 2x3 B.
    # And "h" in 2x2 Strassen maps to "i" (index 4) in 2x3 B.
    
    # p1 = (a+d)(e+h)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[1,0]=1; W2[2,0]=1; W2[3,0]=1

    # p2 = (c+d)e
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[3,1]=-1

    # p3 = a(f-h)  [Original was a(f-g), g->h]
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[2,2]=1; W2[3,2]=1

    # p4 = d(i-e)  [Original was d(h-e), h->i]
    U2[3,3]=1
    V2[4,3]=1; V2[0,3]=-1
    W2[0,3]=-1; W2[1,3]=1

    # p5 = (a+b)h  [Original was (a+b)g, g->h]
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1

    # p6 = (c+a)(i+f) [Original was (c+a)(h+f), h->i]
    U2[2,5]=1; U2[0,5]=1
    V2[4,5]=1; V2[1,5]=1
    W2[3,5]=-1

    # p7 = (b+d)(h-e) [Original was (b+d)(g-e), g->h]
    U2[1,6]=1; U2[3,6]=1
    V2[3,6]=1; V2[0,6]=-1
    W2[0,6]=1

    # Part 2: Schoolbook for B[:,2] = [[g],[j]] (col index 2 and 5 in V, col index 4 and 5 in C)
    # C02 = ag + bj
    # C12 = cg + dj
    
    # ag
    U2[0,7]=1
    V2[2,7]=1
    W2[4,7]=1

    # bj
    U2[1,8]=1
    V2[5,8]=1
    W2[4,8]=1

    # cg
    U2[2,9]=1
    V2[2,9]=1
    W2[5,9]=1

    # dj
    U2[3,10]=1
    V2[5,10]=1
    W2[5,10]=1

    # --- Assemble 4x4x6 from 8 blocks ---
    # Total Rank = 8 * 11 = 88
    # U_total shape: (16, 88)
    # V_total shape: (24, 88)
    # W_total shape: (24, 88)
    
    U_total = np.zeros((16, 88), dtype=float)
    V_total = np.zeros((24, 88), dtype=float)
    W_total = np.zeros((24, 88), dtype=float)

    # Block definitions
    # A blocks (2x2):
    # A00: rows 0,1; cols 0,1 -> U rows 0,1,2,3
    # A01: rows 0,1; cols 2,3 -> U rows 4,5,6,7
    # A10: rows 2,3; cols 0,1 -> U rows 8,9,10,11
    # A11: rows 2,3; cols 2,3 -> U rows 12,13,14,15
    
    # B blocks (2x3):
    # B00: rows 0,1; cols 0,1,2 -> V rows 0,1,2,3,4,5
    # B01: rows 0,1; cols 3,4,5 -> V rows 6,7,8,9,10,11
    # B10: rows 2,3; cols 0,1,2 -> V rows 12,13,14,15,16,17
    # B11: rows 2,3; cols 3,4,5 -> V rows 18,19,20,21,22,23

    # C blocks (2x3):
    # C00: rows 0,1; cols 0,1,2 -> W rows 0,1,2,3,4,5
    # C01: rows 0,1; cols 3,4,5 -> W rows 6,7,8,9,10,11
    # C10: rows 2,3; cols 0,1,2 -> W rows 12,13,14,15,16,17
    # C11: rows 2,3; cols 3,4,5 -> W rows 18,19,20,21,22,23

    # Mapping:
    # A00 * B00 -> C00
    # A01 * B10 -> C00
    # A00 * B01 -> C01
    # A01 * B11 -> C01
    # A10 * B00 -> C10
    # A11 * B10 -> C10
    # A10 * B01 -> C11
    # A11 * B11 -> C11

    blocks = [
        # (A_block_row_offset, B_block_row_offset, C_block_row_offset)
        (0, 0, 0),   # A00, B00 -> C00
        (4, 12, 0),  # A01, B10 -> C00
        (0, 6, 6),   # A00, B01 -> C01
        (4, 18, 6),  # A01, B11 -> C01
        (8, 0, 12),  # A10, B00 -> C10
        (12, 12, 12),# A11, B10 -> C10
        (8, 6, 18),  # A10, B01 -> C11
        (12, 18, 18) # A11, B11 -> C11
    ]

    col_offset = 0
    for u_off, v_off, w_off in blocks:
        U_total[u_off:u_off+4, col_offset:col_offset+11] = U2
        V_total[v_off:v_off+6, col_offset:col_offset+11] = V2
        W_total[w_off:w_off+6, col_offset:col_offset+11] = W2
        col_offset += 11

    if time.time() > deadline:
        return _schoolbook()

    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

def _schoolbook():
    n, m, p = DIMENSIONS
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

# EVOLVE-BLOCK-END
