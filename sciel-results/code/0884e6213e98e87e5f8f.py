"""
Optimized 3x4x6 matrix multiplication tensor decomposition.

Strategy:
  Split the inner dimension m=4 into two blocks of 2.
  For each 3x2x6 sub-problem, split the first dimension n=3 into 1+2:
    - 1x2x6: schoolbook, rank 12
    - 2x2x6: split p=6 into three 2s, use Strassen (rank 7) for each 2x2x2,
             total rank 21
  So each 3x2x6 block uses rank 33.
  Total rank = 33 + 33 = 66.

The output factor W follows the convention W[k*n+i, r] where n=3, i in 0..2,
k in 0..5.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 110  # safety margin before 120s kill

    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)

    # Start with schoolbook rank 72 and refine
    # Actually, let's just build the rank-66 decomposition directly.

    # Strassen decomposition for 2x2x2 (rank 7)
    # U_S[i*2+j], V_S[j*2+k], W_S[k*2+i]
    # The 7 Strassen terms:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # Output:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # For each 2x2x2 block, the U coefficients for A (2x2):
    # M0: A00=1, A11=1  -> [1,0,0,1]
    # M1: A10=1, A11=1  -> [0,0,1,1]
    # M2: A00=1         -> [1,0,0,0]
    # M3: A11=1         -> [0,0,0,1]
    # M4: A00=1, A01=1  -> [1,1,0,0]
    # M5: A10=1, A00=-1 -> [-1,0,1,0]
    # M6: A01=1, A11=-1 -> [0,1,-1,0]

    # V coefficients for B (2x2):
    # M0: B00=1, B11=1  -> [1,0,0,1]
    # M1: B00=1         -> [1,0,0,0]
    # M2: B01=1, B11=-1 -> [0,1,-1,0]
    # M3: B10=1, B00=-1 -> [-1,0,1,0]
    # M4: B11=1         -> [0,0,0,1]
    # M5: B00=1, B01=1  -> [1,1,0,0]
    # M6: B10=1, B11=1  -> [0,0,1,1]

    # W coefficients for C (2x2), C index = k*2+i:
    # M0: C00=1, C11=1  -> [1,0,0,1]
    # M1: C10=1         -> [0,0,1,0]
    # M2: C01=1         -> [0,1,0,0]
    # M3: C00=1, C10=1  -> [1,0,1,0]
    # M4: C00=-1, C01=1 -> [-1,1,0,0]
    # M5: C11=1         -> [0,0,0,1]
    # M6: C00=1, C01=-1 -> [1,-1,0,0]

    U_strassen_222 = [
        [1, 0, 0, 1],   # M0
        [0, 0, 1, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 0, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 1, -1, 0],  # M6
    ]
    V_strassen_222 = [
        [1, 0, 0, 1],   # M0
        [1, 0, 0, 0],   # M1
        [0, 1, -1, 0],  # M2
        [-1, 0, 1, 0],  # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [0, 0, 1, 1],   # M6
    ]
    W_strassen_222 = [
        [1, 0, 0, 1],    # M0: C00=1, C11=1
        [0, 0, 1, 0],    # M1: C10=1
        [0, 1, 0, 0],    # M2: C01=1
        [1, 0, 1, 0],    # M3: C00=1, C10=1
        [-1, 1, 0, 0],   # M4: C00=-1, C01=1
        [0, 0, 0, 1],    # M5: C11=1
        [1, -1, 0, 0],   # M6: C00=1, C01=-1
    ]

    # Now build the rank-66 decomposition for 3x4x6.
    # m=4 split into j-blocks: j_block=0 covers j=0,1; j_block=1 covers j=2,3.
    # For each j_block, we have a 3x2x6 problem.
    # For each 3x2x6, split n=3 into i_block=0 (i=0) and i_block=1 (i=1,2).
    #   - i_block=0: 1x2x6, schoolbook rank 12
    #   - i_block=1: 2x2x6, split p=6 into k_blocks 0,1,2 (k=0,1 / 2,3 / 4,5)
    #     Each 2x2x2 uses Strassen rank 7, total 21.

    # Total rank per j_block: 12 + 21 = 33
    # Total rank: 33 * 2 = 66

    rank = 66
    U = [[0] * rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * rank for _ in range(p * n)]  # 18 x 66

    col = 0

    for j_block in range(2):
        # j_block 0: j = 0, 1; j_block 1: j = 2, 3
        j_offset = j_block * 2

        # --- Part 1: i_block=0 (i=0), 1x2x6 schoolbook, rank 12 ---
        for j_local in range(2):  # j_local 0,1 -> actual j = j_offset + j_local
            j_actual = j_offset + j_local
            i_actual = 0
            for k in range(6):
                # U index: i*m + j = 0*4 + j_actual
                U[i_actual * m + j_actual][col] = 1
                # V index: j*p + k
                V[j_actual * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i_actual][col] = 1
                col += 1
        # col should be at 12 + j_block*12 = 12 or 24

        # --- Part 2: i_block=1 (i=1,2), 2x2x6 via 3 Strassen 2x2x2 blocks, rank 21 ---
        for k_block in range(3):
            k_offset = k_block * 2
            for s in range(7):  # 7 Strassen terms
                # U for A (2x2) with rows i=1,2 and cols j=j_offset, j_offset+1
                # A index in U: i*m + j
                u_vec = U_strassen_222[s]  # [A00, A01, A10, A11]
                # A00 -> i=1, j=j_offset
                # A01 -> i=1, j=j_offset+1
                # A10 -> i=2, j=j_offset
                # A11 -> i=2, j=j_offset+1
                if u_vec[0] != 0:
                    U[1 * m + j_offset][col] += u_vec[0]
                if u_vec[1] != 0:
                    U[1 * m + (j_offset + 1)][col] += u_vec[1]
                if u_vec[2] != 0:
                    U[2 * m + j_offset][col] += u_vec[2]
                if u_vec[3] != 0:
                    U[2 * m + (j_offset + 1)][col] += u_vec[3]

                # V for B (2x2) with rows j=j_offset, j_offset+1 and cols k=k_offset, k_offset+1
                v_vec = V_strassen_222[s]  # [B00, B01, B10, B11]
                if v_vec[0] != 0:
                    V[j_offset * p + k_offset][col] += v_vec[0]
                if v_vec[1] != 0:
                    V[j_offset * p + (k_offset + 1)][col] += v_vec[1]
                if v_vec[2] != 0:
                    V[(j_offset + 1) * p + k_offset][col] += v_vec[2]
                if v_vec[3] != 0:
                    V[(j_offset + 1) * p + (k_offset + 1)][col] += v_vec[3]

                # W for C (2x2) with rows k=k_offset, k_offset+1 and cols i=1,2
                # C index: k*n + i
                w_vec = W_strassen_222[s]  # [C00, C01, C10, C11]
                # C00 -> k=k_offset, i=1
                # C01 -> k=k_offset, i=2
                # C10 -> k=k_offset+1, i=1
                # C11 -> k=k_offset+1, i=2
                if w_vec[0] != 0:
                    W[k_offset * n + 1][col] += w_vec[0]
                if w_vec[1] != 0:
                    W[k_offset * n + 2][col] += w_vec[1]
                if w_vec[2] != 0:
                    W[(k_offset + 1) * n + 1][col] += w_vec[2]
                if w_vec[3] != 0:
                    W[(k_offset + 1) * n + 2][col] += w_vec[3]

                col += 1

    # col should be 66
    # Verify: 2 j_blocks * (12 + 3*7) = 2 * 33 = 66. Yes.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
