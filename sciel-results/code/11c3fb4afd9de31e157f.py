"""Matrix multiplication 6x6x6: Strassen(2x2) x schoolbook(3x3) block decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-level Strassen decomposition:
    - Partition 6x6 matrices into 3x3 blocks of 2x2 matrices.
    - Apply Strassen (rank 7) at the 2x2 block level.
    - Within each 2x2 block, use schoolbook (rank 4) for the 3x3 sub-blocks.
    - Total rank = 7 * 3 * 3 * 4 = 252? No, let's think carefully.

    Actually: A is 6x6, partitioned into 3x3 blocks where each block is 2x2.
    B is 6x6, partitioned into 3x3 blocks where each block is 2x2.
    C = A*B is 6x6, partitioned into 3x3 blocks where each block is 2x2.

    At the block level (3x3 matrix multiplication), we use schoolbook: rank = 3*3*3 = 27.
    For each of the 27 block multiplications, we multiply two 2x2 matrices.
    Using Strassen for 2x2: rank 7 instead of 4.

    So total rank = 27 * 7 = 189.

    Let's verify the indexing:
    A has rows indexed by (i_block, i_local) where i_block in {0,1,2}, i_local in {0,1}
    A has columns indexed by (j_block, j_local) where j_block in {0,1,2}, j_local in {0,1}
    Row index = i_block * 2 + i_local, Col index = j_block * 2 + j_local

    B has rows indexed by (j_block, j_local), cols by (k_block, k_local)
    Row index = j_block * 2 + j_local, Col index = k_block * 2 + k_local

    C has rows indexed by (i_block, i_local), cols by (k_block, k_local)
    Row index = i_block * 2 + i_local, Col index = k_block * 2 + k_local

    The output factor W follows convention W[k*n+i, r] where n=6, p=6.
    So W row index = k*6 + i, where i is the row of C (i_block*2+i_local) and k is the col of C (k_block*2+k_local).
    W[k*6+i, r] = W[k_block*2+k_local, i_block*2+i_local], r]

    For schoolbook at block level: for each (i_block, j_block, k_block), we have a term.
    For Strassen at 2x2 level: 7 multiplications per block pair.

    Strassen formulas for 2x2:
    Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    M1 = (a11+a22)*(b11+b22)
    M2 = (a21+a22)*b11
    M3 = a11*(b12-b22)
    M4 = a22*(b21-b11)
    M5 = (a11+a12)*b22
    M6 = (a21-a11)*(b11+b12)
    M7 = (a12-a22)*(b21+b22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    So for each (i_block, j_block, k_block) triple, we have 7 Strassen multiplications.
    Each Strassen multiplication involves:
    - A linear combination of A's 2x2 block (i_block, j_block)
    - A linear combination of B's 2x2 block (j_block, k_block)

    The A factor for each Strassen term is a 2x2 matrix with entries in {-1, 0, 1}.
    The B factor similarly.

    Then the result contributes to C's 2x2 block (i_block, k_block) with coefficients.

    Let's build U, V, W matrices.

    U has shape (n*m, rank) = (36, rank)
    V has shape (m*p, rank) = (36, rank)
    W has shape (p*n, rank) = (36, rank)

    n=6, m=6, p=6.

    U row index: i*m + j = i*6 + j, where i in 0..5, j in 0..5
    V row index: j*p + k = j*6 + k, where j in 0..5, k in 0..5
    W row index: k*n + i = k*6 + i, where k in 0..5, i in 0..5

    For block (i_block, j_block, k_block):
      i_block in 0..2, j_block in 0..2, k_block in 0..2
      i_local in 0..1, j_local in 0..1, k_local in 0..1

    A element at row i=i_block*2+i_local, col j=j_block*2+j_local
    B element at row j=j_block*2+j_local, col k=k_block*2+k_local
    C element at row i=i_block*2+i_local, col k=k_block*2+k_local

    For Strassen term s in 0..6 for block triple (i_block, j_block, k_block):
      We need A_factor[s] (2x2) and B_factor[s] (2x2) such that:
      A_block(i_block,j_block) * A_factor[s] gives the effective A input
      B_block(j_block,k_block) * B_factor[s] gives the effective B input

      Actually, Strassen computes:
      M_s = (sum over (il,jl) of A[il,jl] * Af[s][il,jl]) * (sum over (jl,kl) of B[jl,kl] * Bf[s][jl,kl])

      Then C[il,kl] += Cf[s][il,kl] * M_s

    So for each Strassen term s:
      U contribution: for each (il, jl), U[i_block*2+il, j_block*2+jl] += Af[s][il,jl]
      V contribution: for each (jl, kl), V[j_block*2+jl, k_block*2+kl] += Bf[s][jl,kl]
      W contribution: for each (il, kl), W[k_block*2+kl, i_block*2+il] += Cf[s][il,kl]

    Let's define the Strassen factors:
    """
    n, m, p = DIMENSIONS  # all 6

    # Strassen 2x2 factors: Af[s][il][jl], Bf[s][jl][kl], Cf[s][il][kl]
    # s=0: M1 = (a11+a22)*(b11+b22)
    # Af[0] = [[1,0],[0,1]]  (a11 + a22)
    # Bf[0] = [[1,0],[0,1]]  (b11 + b22)
    # Cf[0]: C11 += M1, C22 += M1
    # Cf[0] = [[1,0],[0,1]]

    # s=1: M2 = (a21+a22)*b11
    # Af[1] = [[0,0],[1,1]]
    # Bf[1] = [[1,0],[0,0]]
    # Cf[1]: C21 += M2, C22 -= M2
    # Cf[1] = [[0,0],[1,-1]]

    # s=2: M3 = a11*(b12-b22)
    # Af[2] = [[1,0],[0,0]]
    # Bf[2] = [[0,1],[0,-1]]
    # Cf[2]: C12 += M3, C22 += M3
    # Cf[2] = [[0,1],[0,1]]

    # s=3: M4 = a22*(b21-b11)
    # Af[3] = [[0,0],[0,1]]
    # Bf[3] = [[-1,0],[1,0]]
    # Cf[3]: C11 += M4, C21 += M4
    # Cf[3] = [[1,0],[1,0]]

    # s=4: M5 = (a11+a12)*b22
    # Af[4] = [[1,1],[0,0]]
    # Bf[4] = [[0,0],[0,1]]
    # Cf[4]: C11 -= M5, C12 += M5
    # Cf[4] = [[-1,1],[0,0]]

    # s=5: M6 = (a21-a11)*(b11+b12)
    # Af[5] = [[-1,0],[1,0]]
    # Bf[5] = [[1,1],[0,0]]
    # Cf[5]: C22 += M6
    # Cf[5] = [[0,0],[0,1]]

    # s=6: M7 = (a12-a22)*(b21+b22)
    # Af[6] = [[0,1],[0,-1]]
    # Bf[6] = [[0,0],[1,1]]
    # Cf[6]: C11 += M7
    # Cf[6] = [[1,0],[0,0]]

    Af = [
        [[1,0],[0,1]],   # s=0
        [[0,0],[1,1]],   # s=1
        [[1,0],[0,0]],   # s=2
        [[0,0],[0,1]],   # s=3
        [[1,1],[0,0]],   # s=4
        [[-1,0],[1,0]],  # s=5
        [[0,1],[0,-1]],  # s=6
    ]
    Bf = [
        [[1,0],[0,1]],   # s=0
        [[1,0],[0,0]],   # s=1
        [[0,1],[0,-1]],  # s=2
        [[-1,0],[1,0]],  # s=3
        [[0,0],[0,1]],   # s=4
        [[1,1],[0,0]],   # s=5
        [[0,0],[1,1]],   # s=6
    ]
    Cf = [
        [[1,0],[0,1]],   # s=0
        [[0,0],[1,-1]],  # s=1
        [[0,1],[0,1]],   # s=2
        [[1,0],[1,0]],   # s=3
        [[-1,1],[0,0]],  # s=4
        [[0,0],[0,1]],   # s=5
        [[1,0],[0,0]],   # s=6
    ]

    rank = 27 * 7  # 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                for s in range(7):
                    # U: A factor
                    for il in range(2):
                        for jl in range(2):
                            i = i_block * 2 + il
                            j = j_block * 2 + jl
                            U[i * m + j][col] += Af[s][il][jl]
                    # V: B factor
                    for jl in range(2):
                        for kl in range(2):
                            j = j_block * 2 + jl
                            k = k_block * 2 + kl
                            V[j * p + k][col] += Bf[s][jl][kl]
                    # W: C factor, W[k*n+i, col]
                    for il in range(2):
                        for kl in range(2):
                            i = i_block * 2 + il
                            k = k_block * 2 + kl
                            W[k * n + i][col] += Cf[s][il][kl]
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
