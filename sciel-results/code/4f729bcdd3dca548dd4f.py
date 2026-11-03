import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen decomposition of a 2x2 * 2x2 product (7 scalar multiplications)
    # A local indices: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    # B local indices: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    # C local indices: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    strassen_steps = [
        # M1 = (A00+A11)(B00+B11)
        {'u': [(0, 1.0), (3, 1.0)], 'v': [(0, 1.0), (3, 1.0)], 'w': [(0, 1.0), (3, 1.0)]},
        # M2 = (A10+A11)*B00
        {'u': [(2, 1.0), (3, 1.0)], 'v': [(0, 1.0)], 'w': [(2, 1.0), (3, -1.0)]},
        # M3 = A00*(B01-B11)
        {'u': [(0, 1.0)], 'v': [(1, 1.0), (3, -1.0)], 'w': [(1, 1.0), (3, 1.0)]},
        # M4 = A11*(B10-B00)
        {'u': [(3, 1.0)], 'v': [(2, 1.0), (0, -1.0)], 'w': [(0, 1.0), (2, 1.0)]},
        # M5 = (A00+A01)*B11
        {'u': [(0, 1.0), (1, 1.0)], 'v': [(3, 1.0)], 'w': [(0, -1.0), (1, 1.0)]},
        # M6 = (A10-A00)*(B00+B01)
        {'u': [(2, 1.0), (0, -1.0)], 'v': [(0, 1.0), (1, 1.0)], 'w': [(3, 1.0)]},
        # M7 = (A01-A11)*(B10+B11)
        {'u': [(1, 1.0), (3, -1.0)], 'v': [(2, 1.0), (3, 1.0)], 'w': [(0, 1.0)]},
    ]

    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u']:
                lr, lc = local_idx // 2, local_idx % 2
                gi, gj = lr, a_col_start + lc
                U[gi * m + gj][col] += coeff
            for local_idx, coeff in step['v']:
                lr, lc = local_idx // 2, local_idx % 2
                gi, gj = b_row_start + lr, b_col_start + lc
                V[gi * p + gj][col] += coeff
            for local_idx, coeff in step['w']:
                lr, lc = local_idx // 2, local_idx % 2
                gk, gi = c_col_start + lc, lr
                W[gk * n + gi][col] += coeff

    # Four 2x2 blocks of the 2x4 * 4x4 product
    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)

    # Fast verification via matrix multiplication (rank-28 decomposition check)
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)

    # Expected tensor T[i,j,k] = 1 iff A[i] * B[j] -> C[k]
    T_expected = np.zeros((n * m, m * p, p * n))
    for ar in range(n):
        for ac in range(m):
            for bc in range(p):
                i = ar * m + ac
                j = ac * p + bc
                k = bc * n + ar
                T_expected[i, j, k] = 1.0

    T_decomp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    if not np.allclose(T_decomp, T_expected, atol=1e-9):
        # Naive fallback (rank 32)
        rank_naive = n * m * p
        U_f = [[0.0] * rank_naive for _ in range(n * m)]
        V_f = [[0.0] * rank_naive for _ in range(m * p)]
        W_f = [[0.0] * rank_naive for _ in range(p * n)]
        r = 0
        for ar in range(n):
            for ac in range(m):
                for bc in range(p):
                    i = ar * m + ac
                    j = ac * p + bc
                    k = bc * n + ar
                    U_f[i][r] = 1.0
                    V_f[j][r] = 1.0
                    W_f[k][r] = 1.0
                    r += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
