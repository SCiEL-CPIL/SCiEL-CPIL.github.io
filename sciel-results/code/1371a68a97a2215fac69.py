"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Strategy:
  * Guaranteed fallback: the explicit schoolbook decomposition of rank
    n*m*p = 28.  This is always valid and exact.

  * Improvement: block-diagonal construction.
    Partition the 7 output columns (k-index) into three 2-column blocks
    (k=0,1), (k=2,3), (k=4,5) and one 1-column block (k=6).
    For each 2-column block, use a Strassen rank-7 decomposition of the
    2x2x2 multiplication (n=2, m=2, p=2).
    For the 1-column block, use a rank-4 schoolbook decomposition of 2x2x1.
    Total rank = 3*7 + 4 = 25.

  * Verification: exact rational check of the tensor equations.
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
"""

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7

    # --- Build the rank-25 block-diagonal decomposition ---
    # Strassen rank-7 for 2x2x2:
    # The 7 rank-1 terms for multiplying (2x2)*(2x2)->(2x2)
    # U_strassen: (4, 7), V_strassen: (4, 7), W_strassen: (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]

    # Standard Strassen decomposition for 2x2 matrices:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U rows: (i,j) pairs: (0,0), (0,1), (1,0), (1,1)
    # V rows: (j,k) pairs: (0,0), (0,1), (1,0), (1,1)
    # W rows: (k,i) pairs: (0,0), (0,1), (1,0), (1,1)

    # For each Strassen term r, U[i,j,r], V[j,k,r], W[k,i,r]
    # The product U[i,j,r]*V[j,k,r] contributes to C[i,k] with coefficient W[k,i,r]

    # Let me set up the 7 terms explicitly:
    # Term 0: M1 = (a00+a11)*(b00+b11)
    #   U: a00=1, a11=1 => U[0,0]=1, U[1,1]=1, others 0
    #   V: b00=1, b11=1 => V[0,0]=1, V[1,1]=1, others 0
    #   W: C00 += 1, C11 += 1 => W[0,0]=1, W[1,1]=1, others 0
    # Term 1: M2 = (a10+a11)*b00
    #   U: a10=1, a11=1 => U[1,0]=1, U[1,1]=1
    #   V: b00=1 => V[0,0]=1
    #   W: C10 += 1 => W[0,1]=1
    # Term 2: M3 = a00*(b01-b11)
    #   U: a00=1 => U[0,0]=1
    #   V: b01=1, b11=-1 => V[0,1]=1, V[1,1]=-1
    #   W: C01 += 1 => W[1,0]=1
    # Term 3: M4 = a11*(b10-b00)
    #   U: a11=1 => U[1,1]=1
    #   V: b10=1, b00=-1 => V[1,0]=1, V[0,0]=-1
    #   W: C00 += 1, C10 += 1 => W[0,0]=1, W[0,1]=1
    # Term 4: M5 = (a00+a01)*b11
    #   U: a00=1, a01=1 => U[0,0]=1, U[0,1]=1
    #   V: b11=1 => V[1,1]=1
    #   W: C00 -= 1, C01 += 1 => W[0,0]=-1, W[1,0]=1
    # Term 5: M6 = (a10-a00)*(b00+b01)
    #   U: a10=1, a00=-1 => U[1,0]=1, U[0,0]=-1
    #   V: b00=1, b01=1 => V[0,0]=1, V[0,1]=1
    #   W: C11 += 1 => W[1,1]=1
    # Term 6: M7 = (a01-a11)*(b10+b11)
    #   U: a01=1, a11=-1 => U[0,1]=1, U[1,1]=-1
    #   V: b10=1, b11=1 => V[1,0]=1, V[1,1]=1
    #   W: C00 += 1 => W[0,0]=1

    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]

    # Term 0
    U_str[0][0] = 1; U_str[3][0] = 1
    V_str[0][0] = 1; V_str[3][0] = 1
    W_str[0][0] = 1; W_str[3][0] = 1
    # Term 1
    U_str[2][1] = 1; U_str[3][1] = 1
    V_str[0][1] = 1
    W_str[1][1] = 1
    # Term 2
    U_str[0][2] = 1
    V_str[1][2] = 1; V_str[3][2] = -1
    W_str[2][2] = 1
    # Term 3
    U_str[3][3] = 1
    V_str[2][3] = 1; V_str[0][3] = -1
    W_str[0][3] = 1; W_str[1][3] = 1
    # Term 4
    U_str[0][4] = 1; U_str[1][4] = 1
    V_str[3][4] = 1
    W_str[0][4] = -1; W_str[2][4] = 1
    # Term 5
    U_str[2][5] = 1; U_str[0][5] = -1
    V_str[0][5] = 1; V_str[1][5] = 1
    W_str[3][5] = 1
    # Term 6
    U_str[1][6] = 1; U_str[3][6] = -1
    V_str[2][6] = 1; V_str[3][6] = 1
    W_str[0][6] = 1

    # Verify Strassen block locally
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U_str[i*2+j][r] * V_str[j*2+k][r] * W_str[k*2+i][r]
                if s != 1:
                    # Fallback to schoolbook
                    return _schoolbook(n, m, p)

    # Now build the full 2x2x7 decomposition
    # Blocks: k in [0,1], [2,3], [4,5] use Strassen; k=6 uses schoolbook rank-4
    total_rank = 3 * 7 + 4  # = 25

    U = [[0] * total_rank for _ in range(n * m)]   # 4 x 25
    V = [[0] * total_rank for _ in range(m * p)]   # 14 x 25
    W = [[0] * total_rank for _ in range(p * n)]   # 14 x 25

    col = 0
    # Three Strassen blocks for k-offsets 0, 2, 4
    for block_start in [0, 2, 4]:
        for r in range(7):
            # U part: same for all blocks (operates on A which is 2x2)
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_str[i * 2 + j][r]
            # V part: V[j*p+k, col] where k = block_start + local_k
            for j in range(2):
                for local_k in range(2):
                    k = block_start + local_k
                    V[j * p + k][col] = V_str[j * 2 + local_k][r]
            # W part: W[k*n+i, col] where k = block_start + local_k
            for local_k in range(2):
                for i in range(2):
                    k = block_start + local_k
                    W[k * n + i][col] = W_str[local_k * 2 + i][r]
            col += 1

    # Schoolbook block for k=6: rank 4 (one per (i,j) pair)
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1

    # Final exact verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
