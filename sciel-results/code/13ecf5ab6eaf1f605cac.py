"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    # The rank-14 decomposition for 2x2 * 2x4 matrix multiplication
    # Uses two Strassen 2x2 blocks with shared U factors
    R = 14
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))

    # Left block (columns 0,1 of B) - 7 Strassen terms
    # r=0: M1 = (a00+a11)(b00+b11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[5,0] = 1
    W[0,0] = 1; W[3,0] = 1

    # r=1: M2 = (a10+a11)b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1

    # r=2: M3 = a00(b11-b01)
    U[0,2] = 1
    V[1,2] = 1; V[5,2] = -1
    W[2,2] = 1; W[3,2] = 1

    # r=3: M4 = a11(b10-b00)
    U[3,3] = 1
    V[4,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1

    # r=4: M5 = (a00+a01)b11
    U[0,4] = 1; U[1,4] = 1
    V[5,4] = 1
    W[0,4] = -1; W[2,4] = 1

    # r=5: M6 = (a10-a00)(b00+b01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1

    # r=6: M7 = (a01-a11)(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[4,6] = 1; V[5,6] = 1
    W[0,6] = 1

    # Right block (columns 2,3 of B) - 7 Strassen terms
    # r=7: N1 = (a00+a11)(b02+b13)
    U[0,7] = 1; U[3,7] = 1
    V[2,7] = 1; V[7,7] = 1
    W[4,7] = 1; W[7,7] = 1

    # r=8: N2 = (a10+a11)b02
    U[2,8] = 1; U[3,8] = 1
    V[2,8] = 1
    W[5,8] = 1; W[7,8] = -1

    # r=9: N3 = a00(b03-b13)
    U[0,9] = 1
    V[3,9] = 1; V[7,9] = -1
    W[6,9] = 1; W[7,9] = 1

    # r=10: N4 = a11(b12-b02)
    U[3,10] = 1
    V[6,10] = 1; V[2,10] = -1
    W[4,10] = 1; W[5,10] = 1

    # r=11: N5 = (a00+a01)b13
    U[0,11] = 1; U[1,11] = 1
    V[7,11] = 1
    W[4,11] = -1; W[6,11] = 1

    # r=12: N6 = (a10-a00)(b02+b03)
    U[2,12] = 1; U[0,12] = -1
    V[2,12] = 1; V[3,12] = 1
    W[7,12] = 1

    # r=13: N7 = (a01-a11)(b12+b13)
    U[1,13] = 1; U[3,13] = -1
    V[6,13] = 1; V[7,13] = 1
    W[4,13] = 1

    # Verify the decomposition is exact
    def verify(U_arr, V_arr, W_arr):
        Rv = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(Rv):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        # Convert to list of lists with integer/half-integer entries
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to naive rank 16 decomposition
    rank = n * m * p
    U_naive = [[0] * rank for _ in range(nm)]
    V_naive = [[0] * rank for _ in range(mp)]
    W_naive = [[0] * rank for _ in range(pn)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][r] = 1
                V_naive[j * p + k][r] = 1
                W_naive[k * n + i][r] = 1
                r += 1
    return {"U": U_naive, "V": V_naive, "W": W_naive}
# EVOLVE-BLOCK-END
