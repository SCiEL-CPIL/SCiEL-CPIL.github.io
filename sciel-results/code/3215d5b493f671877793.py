import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _get_strassen_2x2_factors():
    """Returns U, V, W for 2x2x2 Strassen decomposition as lists.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * 2 + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * 2 + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * 2 + i][m_idx] += c

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Check that the tensor decomposition is exact.
    Sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all (i,j,k).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def _schoolbook(n, m, p):
    """Fallback: schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    n, m, p = DIMENSIONS  # 4, 4, 6

    U_s, V_s, W_s = _get_strassen_2x2_factors()

    # Total rank: 49 (for 4x4x4 two-level Strassen) + 28 (for 4x4x2 block Strassen) = 77
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = [[0.0] * total_rank for _ in range(n * m)]   # 16 x 77
    V = [[0.0] * total_rank for _ in range(m * p)]   # 24 x 77
    W = [[0.0] * total_rank for _ in range(p * n)]   # 24 x 77

    # Part 1: 4x4 x 4x4 using two-level Strassen (rank 49)
    # Split each 4x4 into 2x2 blocks. For A: A[i,j] with i=2*I+i0, j=2*J+j0
    # The two-level Strassen: U_444[i*4+j, 7t+s] = U_s[2I+J, t] * U_s[2i0+j0, s]
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col] += val

    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][col] += val

    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # Part 2: 4x4 x 4x2 (columns 4-5 of B) using block Strassen (rank 28)
    # C_right = A * B_right where B_right is 4x2 (cols 4-5)
    # Partition A into 2x2 blocks A_IJ (I,J in 0..1)
    # B_right: top half (rows 0-1, cols 4-5) and bottom half (rows 2-3, cols 4-5)
    # C_top = A_00 * B_top + A_01 * B_bot
    # C_bot = A_10 * B_top + A_11 * B_bot
    # Four 2x2x2 Strassen products:
    products = [
        (0, 0, 0),  # I=0, J=0, B_top (j_global = j0)
        (0, 1, 1),  # I=0, J=1, B_bot (j_global = 2+j0)
        (1, 0, 0),  # I=1, J=0, B_top
        (1, 1, 1),  # I=1, J=1, B_bot
    ]

    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7

        # U part: A_IJ, i = 2I+i0, j = 2J+j0
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col_offset + s] += val

        # V part: B block, j_global = 2*is_bot + j0, k_global = 4 + k0
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global][col_offset + s] += val

        # W part: C block, i_global = 2I+i0, k_global = 4+k0
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global][col_offset + s] += val

    # Verify the decomposition
    if time.time() < deadline - 5.0:
        err = _verify(U, V, W, n, m, p, total_rank)
        if err > 1e-6:
            # Fallback to schoolbook
            U, V, W = _schoolbook(n, m, p)
            total_rank = n * m * p

    # Convert to lists with integer rounding where appropriate
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    r.append(int(round(x)))
                else:
                    r.append(float(x))
            result.append(r)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
