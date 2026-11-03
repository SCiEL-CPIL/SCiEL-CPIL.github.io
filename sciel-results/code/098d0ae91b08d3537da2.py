"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-24 exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix.
    
    Strategy:
    1. Split the 3x6 matrix B into three 3x2 blocks B_b for b in {0,1,2}.
    2. For each block, split the multiplication A * B_b into:
       - An outer product part involving A[:, 0] and B_b[0, :] (Rank 1 per block).
       - A 2x2 matrix multiplication involving A[:, 1:3] and B_b[1:3, :] (Rank 7 per block using Strassen).
    3. Total rank = 3 * (1 + 7) = 24.
    """
    n, m, p = DIMENSIONS
    rank = 24
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank) -> (6, 24)
    # V shape: (m*p, rank) -> (18, 24)
    # W shape: (p*n, rank) -> (12, 24)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Part 1: Outer product terms (Rank 3)
    # For each block b, handle j=0
    for b in range(3):
        r = b
        # U[i, 0, r] = 1
        for i in range(n):
            U[i * m + 0][r] = 1.0
        # V[0, 2*b+k', r] = 1
        for kp in range(2):
            V[0 * p + (2 * b + kp)][r] = 1.0
        # W[2*b+k', i, r] = 1
        for kp in range(2):
            for i in range(n):
                W[(2 * b + kp) * n + i][r] = 1.0

    # Part 2: Strassen terms for 2x2 multiplication (Rank 21)
    # Strassen coefficients for M1..M7
    # u_s[i, jp] where jp is 0 or 1 (corresponding to j=1,2)
    # v_s[jp, kp] where jp is 0 or 1, kp is 0 or 1
    # w_s[kp, i] where kp is 0 or 1, i is 0 or 1
    
    strassen_u = [
        [[1, 0], [0, 1]], # M1: a00 + a11
        [[0, 0], [1, 1]], # M2: a10 + a11
        [[1, 0], [0, 0]], # M3: a00
        [[0, 0], [0, 1]], # M4: a11
        [[1, 1], [0, 0]], # M5: a00 + a01
        [[-1, 0], [1, 0]],# M6: a10 - a00
        [[0, 1], [0, -1]] # M7: a01 - a11
    ]
    
    strassen_v = [
        [[1, 0], [0, 1]], # M1: b00 + b11
        [[1, 0], [0, 0]], # M2: b00
        [[0, 0], [1, -1]],# M3: b10 - b11
        [[-1, 1], [0, 0]],# M4: b01 - b00
        [[0, 0], [0, 1]], # M5: b11
        [[1, 0], [1, 0]], # M6: b00 + b10
        [[0, 1], [0, 1]]  # M7: b01 + b11
    ]
    
    strassen_w = [
        [[1, 0], [0, 1]], # M1: c00 + c11 (w[kp, i]) -> w[0,0]=1, w[1,1]=1
        [[0, 1], [-1, 0]],# M2: c10 - c11 -> w[0,1]=1, w[1,1]=-1
        [[0, 0], [1, 1]], # M3: c01 + c11 -> w[1,0]=1, w[1,1]=1
        [[1, 0], [1, 0]], # M4: c00 + c10 -> w[0,0]=1, w[0,1]=1
        [[-1, 0], [1, 0]],# M5: -c00 + c01 -> w[0,0]=-1, w[1,0]=1
        [[0, 0], [0, 1]], # M6: c11 -> w[1,1]=1
        [[1, 0], [0, 0]]  # M7: c00 -> w[0,0]=1
    ]
    
    for b in range(3):
        base_r = 3 + 7 * b
        for s in range(7):
            r = base_r + s
            # Fill U
            for i in range(n):
                for jp in range(2):
                    val = strassen_u[s][i][jp]
                    if val != 0:
                        U[i * m + (1 + jp)][r] = val
            # Fill V
            for jp in range(2):
                for kp in range(2):
                    val = strassen_v[s][jp][kp]
                    if val != 0:
                        V[(1 + jp) * p + (2 * b + kp)][r] = val
            # Fill W
            for kp in range(2):
                for i in range(n):
                    val = strassen_w[s][kp][i]
                    if val != 0:
                        W[(2 * b + kp) * n + i][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
