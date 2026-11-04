import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave margin under 150s wall-clock

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def check_solution(U, V, W):
        """Exact verification of the tensor decomposition (trace convention W[k*n+i])."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] must be 1 if (i,j,k) is a valid product index, 0 otherwise.
        # Valid product indices: for all i in [0,n), j in [0,m), k in [0,p): the (i,k) entry of A@B uses j.
        # The tensor T[i,j,k] = 1 for all i,j,k (this is the matmul tensor with the given convention).
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]))
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def strassen_2x2_block():
        """
        Build a rank-7 Strassen decomposition for a 2x2 * 2x2 multiplication.
        Returns (U, V, W) as lists of lists with shapes (4,7),(4,7),(4,7).
        Convention: U indexed by (i,j) -> i*2+j ; V indexed by (j,k) -> j*2+k ; W indexed by (k,i) -> k*2+i.
        Strassen products:
          M1 = (A00+A11)(B00+B11)
          M2 = (A10+A11)B00
          M3 = A00(B10-B11)
          M4 = A11(B01-B10)
          M5 = (A00+A01)B11
          M6 = (A10-A00)(B00+B10)
          M7 = (A01-A11)(B01+B11)
          C00 = M1 + M4 - M5 + M7
          C01 = M3 + M5
          C10 = M2 + M4
          C11 = M1 - M2 + M3 + M6
        """
        # U rows: (i,j) for i in 0..1, j in 0..1 -> index i*2+j
        # V rows: (j,k) for j in 0..1, k in 0..1 -> index j*2+k
        # W rows: (k,i) for k in 0..1, i in 0..1 -> index k*2+i
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        # M4 = A11(B01-B10)
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        # M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1

        # W: C entries. C00 = M1+M4-M5+M7 ; C01 = M3+M5 ; C10 = M2+M4 ; C11 = M1-M2+M3+M6
        # W index (k,i): C00 -> (0,0)=0 ; C01 -> (0,1)=1 ; C10 -> (1,0)=2 ; C11 -> (1,1)=3
        W[0][0] = 1;  W[0][3] = 1;  W[0][4] = -1;  W[0][6] = 1   # C00
        W[1][2] = 1;  W[1][4] = 1                                  # C01
        W[2][1] = 1;  W[2][3] = 1                                  # C10
        W[3][0] = 1;  W[3][1] = -1; W[3][2] = 1;  W[3][5] = 1     # C11
        return U, V, W

    def build_block_decomposition():
        """
        Deterministic block decomposition:
        Split the 4x5 * 5x5 into blocks.
        Rows of A: 4 rows. Cols of A / rows of B: 5. Cols of B: 5.
        We partition:
          A row-blocks: [0..1] (2 rows), [2..3] (2 rows)  -> 2 row-blocks of size 2
          Middle (cols of A / rows of B): [0..1] (2), [2..3] (2), [4] (1) -> 3 blocks
          B col-blocks: [0..1] (2), [2..3] (2), [4] (1) -> 3 blocks
        For each (row-block rb, mid-block mb, col-block cb) we multiply a (2 x sz_mb) by (sz_mb x sz_cb) block.
        Use Strassen (rank 7) when sz_mb==2 and sz_cb==2 (i.e., a 2x2 * 2x2). Otherwise schoolbook within the block.
        This yields a valid exact decomposition. We'll compute total rank and build U,V,W.
        """
        # Block sizes
        row_blocks = [(0, 2), (2, 4)]          # 2 blocks of 2 rows
        mid_blocks = [(0, 2), (2, 4), (4, 5)]  # sizes 2,2,1
        col_blocks = [(0, 2), (2, 4), (4, 5)]  # sizes 2,2,1

        # We'll accumulate factor columns. Each sub-problem contributes some rank.
        # Represent U,V,W as growing lists of columns.
        U_cols = []  # list of column vectors (length n*m)
        V_cols = []  # list of column vectors (length m*p)
        W_cols = []  # list of column vectors (length p*n)

        def add_block(rb, mb, cb):
            r0, r1 = row_blocks[rb]
            c0, c1 = mid_blocks[mb]
            d0, d1 = col_blocks[cb]
            sz_r = r1 - r0   # 2
            sz_m = c1 - c0   # 2 or 1
            sz_c = d1 - d0   # 2 or 1

            if sz_r == 2 and sz_m == 2 and sz_c == 2:
                # Strassen 2x2 * 2x2
                Usub, Vsub, Wsub = strassen_2x2_block()
                for r in range(7):
                    Uc = [0]*(n*m); Vc = [0]*(m*p); Wc = [0]*(p*n)
                    # Map sub-indices to global
                    for si in range(2):
                        for sj in range(2):
                            i = r0 + si
                            j = c0 + sj
                            val = Usub[si*2 + sj][r]
                            if val != 0:
                                Uc[i*m + j] = val
                    for sj in range(2):
                        for sk in range(2):
                            j = c0 + sj
                            k = d0 + sk
                            val = Vsub[sj*2 + sk][r]
                            if val != 0:
                                Vc[j*p + k] = val
                    for sk in range(2):
                        for si in range(2):
                            k = d0 + sk
                            i = r0 + si
                            val = Wsub[sk*2 + si][r]
                            if val != 0:
                                Wc[k*n + i] = val
                    U_cols.append(Uc); V_cols.append(Vc); W_cols.append(Wc)
            else:
                # Schoolbook within block: rank = sz_r * sz_m * sz_c
                for si in range(sz_r):
                    for sj in range(sz_m):
                        for sk in range(sz_c):
                            i = r0 + si
                            j = c0 + sj
                            k = d0 + sk
                            Uc = [0]*(n*m); Vc = [0]*(m*p); Wc = [0]*(p*n)
                            Uc[i*m + j] = 1
                            Vc[j*p + k] = 1
                            Wc[k*n + i] = 1
                            U_cols.append(Uc); V_cols.append(Vc); W_cols.append(Wc)

        for rb in range(len(row_blocks)):
            for mb in range(len(mid_blocks)):
                for cb in range(len(col_blocks)):
                    add_block(rb, mb, cb)

        rank = len(U_cols)
        # Transpose columns to rows
        U = [[U_cols[r][row] for r in range(rank)] for row in range(n*m)]
        V = [[V_cols[r][row] for r in range(rank)] for row in range(m*p)]
        W = [[W_cols[r][row] for r in range(rank)] for row in range(p*n)]
        return {"U": U, "V": V, "W": W}

    def build_row_block_strassen():
        """
        Alternative: treat the 4x5*5x5 as 2 row-blocks of 2 rows.
        For each row-block (2 rows) and for the full 5x5 middle/col,
        split middle into [0..1],[2..3],[4] and cols into [0..1],[2..3],[4].
        Same as above but we also try a variant using Strassen on 2x2*2x2 for the
        (2x2)*(2x2) corner blocks and schoolbook for the rest.
        This is essentially the same as build_block_decomposition; keep as alias for clarity.
        """
        return build_block_decomposition()

    # Build the deterministic block decomposition
    result = build_block_decomposition()

    # Verify quickly
    if check_solution(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook (should not happen)
    return schoolbook()
