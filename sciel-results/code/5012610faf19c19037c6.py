"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product of Strassen's rank-7 2x2x2 matrix multiplication
    # gives rank 7*7 = 49 for 4x4x4, improving on schoolbook rank 64.
    #
    # Strassen's 7 multiplications for 2x2x2:
    #   M1 = (A11 - A21)(B11 + B21)
    #   M2 = (A11 + A21)B21
    #   M3 = A12(B21 - B11)
    #   M4 = A22(B11 - B21)
    #   M5 = (A11 + A12)B11
    #   M6 = (A21 - A11)(B11 + B12)
    #   M7 = (A12 - A22)(B21 + B22)
    #
    # The 4x4x4 tensor = 2x2x2 tensor (Kronecker) with 2x2 blocks.
    # We build U, V, W of shape (16, 49), (16, 49), (16, 49) where:
    #   U rows indexed by (i, j) with i in [0,3], j in [0,3]  -> row = i*4+j
    #   V rows indexed by (j, k) with j in [0,3], k in [0,3]  -> row = j*4+k
    #   W rows indexed by (k, i) with k in [0,3], i in [0,3]  -> row = k*4+i
    #   Column c corresponds to a pair (a, b) of Strassen indices, each in [0,6],
    #   so c = a*7 + b.

    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2x2:
    # For each multiplication l in 0..6, we have:
    #   U_factor: coefficient for A_{row, col} in the first factor
    #   V_factor: coefficient for B_{row, col} in the second factor
    #   W_factor: which output C_{row, col} gets this multiplication
    #
    # Strassen's 7 multiplications (0-indexed):
    # l=0: M1 = (A11 - A21)(B11 + B21)
    # l=1: M2 = (A11 + A21)B21
    # l=2: M3 = A12(B21 - B11)
    # l=3: M4 = A22(B11 - B21)
    # l=4: M5 = (A11 + A12)B11
    # l=5: M6 = (A21 - A11)(B11 + B12)
    # l=6: M7 = (A12 - A22)(B21 + B22)
    #
    # For the Kronecker product, the 4x4 matrix A is partitioned into 2x2 blocks
    # of 2x2 sub-matrices: A_{block_row, block_col} where block indices are in {0,1}.
    # Within each block, sub-indices are in {0,1}.
    #
    # So A[i,j] where i = 2*bi + si, j = 2*bj + sj (bi,sj in {0,1})
    # The Strassen formula operates at the block level with the same indices.
    #
    # For the Kronecker product construction:
    # For Strassen multiplication l with:
    #   First factor: sum over (ar, ac) of coeff_U_l[ar][ac] * A_{ar, ac}
    #   Second factor: sum over (br, bc) of coeff_V_l[br][bc] * B_{br, bc}
    #   Output: sum over (cr, cc) of coeff_W_l[cr][cc] * C_{cr, cc} = M_l
    #
    # In the Kronecker product, for each pair of Strassen indices (a, b):
    #   U[i*4+j, a*7+b] = coeff_U_a[i_block][j_block] * coeff_U_b[i_sub][j_sub]
    #   V[j*4+k, a*7+b] = coeff_V_a[j_block][k_block] * coeff_V_b[j_sub][k_sub]
    #   W[k*4+i, a*7+b] = coeff_W_a[k_block][i_block] * coeff_W_b[k_sub][i_sub]
    #   where i = 2*i_block + i_sub, j = 2*j_block + j_sub, etc.

    # Define Strassen coefficients as 2x2 matrices
    # U_l: 2x2 matrix of coefficients for A in multiplication l
    # V_l: 2x2 matrix of coefficients for B in multiplication l
    # W_l: 2x2 matrix of coefficients for C in multiplication l (output)

    # l=0: M1 = (A11 - A21)(B11 + B21)
    #   U_0 = [[1, 0], [-1, 0]]  (A11 coeff 1, A21 coeff -1, A12 coeff 0, A22 coeff 0)
    #   V_0 = [[1, 0], [1, 0]]   (B11 coeff 1, B21 coeff 1, B12 coeff 0, B22 coeff 0)
    #   C11 = M1 + M4 - M5 + M7  -> W_0[1][1] += 1
    #   C22 = M1 - M2 + M3 + M6  -> W_0[2][2] += 1  (using 1-indexed blocks: C22 is block (1,1))
    # Wait, let me use 0-indexed: blocks (0,0), (0,1), (1,0), (1,1)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Using 0-indexed block coordinates (row, col):
    # l=0: U_0 = [[1, 0], [-1, 0]], V_0 = [[1, 0], [1, 0]]
    # l=1: U_1 = [[1, 0], [1, 0]],  V_1 = [[0, 0], [0, 1]]
    # l=2: U_2 = [[0, 1], [0, 0]],  V_2 = [[0, -1], [0, 1]]
    # l=3: U_3 = [[0, 0], [0, 1]],  V_3 = [[0, 1], [0, -1]]
    # l=4: U_4 = [[1, 1], [0, 0]],  V_4 = [[1, 0], [0, 0]]
    # l=5: U_5 = [[-1, 0], [1, 0]], V_5 = [[1, 1], [0, 0]]
    # l=6: U_6 = [[0, 1], [0, -1]], V_6 = [[0, 1], [1, 1]]

    U_strassen = [
        [[1, 0], [-1, 0]],   # l=0
        [[1, 0], [1, 0]],    # l=1
        [[0, 1], [0, 0]],    # l=2
        [[0, 0], [0, 1]],    # l=3
        [[1, 1], [0, 0]],    # l=4
        [[-1, 0], [1, 0]],   # l=5
        [[0, 1], [0, -1]],   # l=6
    ]
    V_strassen = [
        [[1, 0], [1, 0]],    # l=0
        [[0, 0], [0, 1]],    # l=1
        [[0, -1], [0, 1]],   # l=2
        [[0, 1], [0, -1]],   # l=3
        [[1, 0], [0, 0]],    # l=4
        [[1, 1], [0, 0]],    # l=5
        [[0, 1], [1, 1]],    # l=6
    ]
    # W coefficients: for each l, which C blocks get contributions
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    W_strassen = [
        [[1, 0], [0, 1]],    # l=0: C00 += 1, C11 += 1
        [[0, 0], [1, -1]],   # l=1: C10 += 1, C11 += -1
        [[0, 1], [0, 1]],    # l=2: C01 += 1, C11 += 1
        [[1, 0], [1, 0]],    # l=3: C00 += 1, C10 += 1
        [[-1, 1], [0, 0]],   # l=4: C00 += -1, C01 += 1
        [[0, 0], [0, 1]],    # l=5: C11 += 1
        [[1, 0], [0, 0]],    # l=6: C00 += 1
    ]

    # Build U, V, W for 4x4x4
    for a in range(7):
        for b in range(7):
            col = a * 7 + b
            for bi in range(2):  # block row of A
                for bj in range(2):  # block col of A
                    for si in range(2):  # sub row
                        for sj in range(2):  # sub col
                            i = 2 * bi + si
                            j = 2 * bj + sj
                            U[i * 4 + j][col] = U_strassen[a][bi][bj] * U_strassen[b][si][sj]

            for bi in range(2):  # block row of B
                for bj in range(2):  # block col of B
                    for sj in range(2):  # sub row (matches A's col)
                        for sk in range(2):  # sub col
                            j = 2 * bj + sj
                            k = 2 * bi + sk
                            V[j * 4 + k][col] = V_strassen[a][bi][bj] * V_strassen[b][sj][sk]

            for bi in range(2):  # block row of C (output)
                for bj in range(2):  # block col of C
                    for si in range(2):  # sub row
                        for sj in range(2):  # sub col
                            k = 2 * bi + si
                            i = 2 * bj + sj
                            W[k * 4 + i][col] = W_strassen[a][bi][bj] * W_strassen[b][si][sj]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
