"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition using Strassen on 2x2x2 sub-blocks
    # Structure:
    # Split p=6 into 3 blocks of size 2.
    # For each block, split m=5 into 2+2+1.
    # Use Strassen (rank 7) for the two 2x2x2 parts.
    # Use schoolbook (rank 4) for the 2x1x2 part.
    # Total rank = 3 * (7 + 7 + 4) = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00+, C11+
    # M2 = (A10+A11)B00       -> C10+, C11-
    # M3 = A00(B10-B11)       -> C01+, C11+
    # M4 = A11(B01-B00)       -> C00+, C10+
    # M5 = (A00+A01)B11       -> C00-, C01+
    # M6 = (A10-A00)(B00+B01) -> C11+
    # M7 = (A01-A11)(B10+B11) -> C00+
    
    strassen_u = [
        [1, 0, 1, 0], # A00, A01, A10, A11 for M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1], # B00, B01, B10, B11 for M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [-1, 1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W is indexed by [k*n + i, rank]
    # Local k in {0,1}, i in {0,1}
    # Indices: 0:(k=0,i=0), 1:(k=0,i=1), 2:(k=1,i=0), 3:(k=1,i=1)
    strassen_w = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, -1],# M2: C10, C11
        [0, 0, 1, 1], # M3: C01, C11
        [1, 1, 0, 0], # M4: C00, C10
        [-1, 0, 1, 0],# M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]

    # Schoolbook for 2x1x2 (Rank 4)
    # A is 2x1 (A00, A10), B is 1x2 (B00, B01)
    # M8 = A00 B00 -> C00
    # M9 = A00 B01 -> C01
    # M10 = A10 B00 -> C10
    # M11 = A10 B01 -> C11
    
    school_u_212 = [
        [1, 0], # A00, A10 for M8
        [1, 0], # M9
        [0, 1], # M10
        [0, 1]  # M11
    ]
    
    school_v_212 = [
        [1, 0], # B00, B01 for M8
        [0, 1], # M9
        [1, 0], # M10
        [0, 1]  # M11
    ]
    
    school_w_212 = [
        [1, 0, 0, 0], # M8: C00
        [0, 0, 1, 0], # M9: C01
        [0, 1, 0, 0], # M10: C10
        [0, 0, 0, 1]  # M11: C11
    ]

    for b in range(3):
        k_start = 2 * b
        
        # Block 1: j in {0, 1}, Strassen
        j_start = 0
        for t in range(7):
            # U
            for li in range(2):
                for lj in range(2):
                    val = strassen_u[t][li * 2 + lj]
                    if val != 0:
                        gi = li
                        gj = j_start + lj
                        U[gi * m + gj][col] = val
            # V
            for lj in range(2):
                for lk in range(2):
                    val = strassen_v[t][lj * 2 + lk]
                    if val != 0:
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk][col] = val
            # W
            for lk in range(2):
                for li in range(2):
                    val = strassen_w[t][lk * 2 + li]
                    if val != 0:
                        gk = k_start + lk
                        gi = li
                        W[gk * n + gi][col] = val
            col += 1

        # Block 2: j in {2, 3}, Strassen
        j_start = 2
        for t in range(7):
            # U
            for li in range(2):
                for lj in range(2):
                    val = strassen_u[t][li * 2 + lj]
                    if val != 0:
                        gi = li
                        gj = j_start + lj
                        U[gi * m + gj][col] = val
            # V
            for lj in range(2):
                for lk in range(2):
                    val = strassen_v[t][lj * 2 + lk]
                    if val != 0:
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk][col] = val
            # W
            for lk in range(2):
                for li in range(2):
                    val = strassen_w[t][lk * 2 + li]
                    if val != 0:
                        gk = k_start + lk
                        gi = li
                        W[gk * n + gi][col] = val
            col += 1

        # Block 3: j in {4}, Schoolbook 2x1x2
        j_start = 4
        for t in range(4):
            # U
            for li in range(2):
                val = school_u_212[t][li]
                if val != 0:
                    gi = li
                    gj = j_start
                    U[gi * m + gj][col] = val
            # V
            for lk in range(2):
                val = school_v_212[t][lk]
                if val != 0:
                    gj = j_start
                    gk = k_start + lk
                    V[gj * p + gk][col] = val
            # W
            for lk in range(2):
                for li in range(2):
                    val = school_w_212[t][lk * 2 + li]
                    if val != 0:
                        gk = k_start + lk
                        gi = li
                        W[gk * n + gi][col] = val
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
