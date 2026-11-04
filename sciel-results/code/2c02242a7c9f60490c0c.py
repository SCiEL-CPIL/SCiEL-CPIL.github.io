"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication (Strassen)."""
        terms = []
        # p1 = (a11+a22)*(b11+b22)
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # p2 = (a21+a22)*b11
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # p3 = a11*(b12-b22)
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # p4 = a22*(b21-b11)
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # p5 = (a11+a12)*b22
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # p6 = (a21-a11)*(b11+b12)
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # p7 = (a12-a22)*(b21+b22)
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    # Build the full 3x4 * 4x4 decomposition.
    # Split strategy:
    #   Rows of A: top 2 (i=0,1), bottom 1 (i=2)
    #   Cols of B / rows of C: left 2 (k=0,1), right 2 (k=2,3)
    #   Middle dim j: j=0,1 and j=2,3
    #
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  -> 7+7=14
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  -> 7+7=14
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]  -> 2+2=4
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]  -> 2+2=4
    # Total: 36

    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 36
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 36
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 36

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, w_row_offset, col_start):
        """Add a rank-7 2x2*2x2 block.
        a_row_offset: row offset in A (0 or 1)
        b_col_offset: col offset in B (0 or 2)
        j_offset: row offset in B / col offset in A (0 or 2)
        w_row_offset: row offset in C (0 or 2)
        """
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            col = col_start + r
            for (ai, aj), val in u_dict.items():
                U[a_row_offset + ai][col] += val
            for (bj, bk), val in v_dict.items():
                V[(j_offset + bj) * p + (b_col_offset + bk)][col] += val
            for (ci, ck), val in w_dict.items():
                W[(w_row_offset + ck) * n + (w_row_offset + ci)][col] += val
        return col_start + 7

    def add_rank2_block_1x2(a_row, b_col_offset, j_offset, col_start):
        """Add a rank-2 1x2*2x2 block (bottom row of A, 2 columns of B)."""
        for r in range(2):
            col = col_start + r
            U[a_row * m + (j_offset + r)][col] = 1
            V[(j_offset + r) * p + b_col_offset][col] = 1
            V[(j_offset + r) * p + b_col_offset + 1][col] = 1
            W[b_col_offset * n + a_row][col] = 1
            W[(b_col_offset + 1) * n + a_row][col] = 1
        return col_start + 2

    col = 0
    col = add_rank7_block(0, 0, 0, 0, col)
    col = add_rank7_block(0, 0, 2, 0, col)
    col = add_rank7_block(0, 2, 0, 0, col)
    col = add_rank7_block(0, 2, 2, 0, col)
    col = add_rank2_block_1x2(2, 0, 0, col)
    col = add_rank2_block_1x2(2, 0, 2, col)
    col = add_rank2_block_1x2(2, 2, 0, col)
    col = add_rank2_block_1x2(2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            a_r = sum(U_arr[i*m+j, r] * A_test[i,j] for i in range(n) for j in range(m))
            b_r = sum(V_arr[j*p+k, r] * B_test[j,k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += W_arr[k*n+i, r] * a_r * b_r

        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-9:
            print(f"Verification failed: max error = {max_err}")
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception as e:
        print(f"Verification error: {e}")
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
