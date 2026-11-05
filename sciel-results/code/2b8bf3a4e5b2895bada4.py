"""Matrix multiplication 5x5x5 with rank-117 decomposition using Strassen on 2x2x2 blocks."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5

    # Partition each dimension into 2+2+1
    parts = [[0, 1], [2, 3], [4]]
    part_sizes = [2, 2, 1]

    U_rows = [dict() for _ in range(n * m)]
    V_rows = [dict() for _ in range(m * p)]
    W_rows = [dict() for _ in range(p * n)]

    col = 0

    for ii in range(3):
        for jj in range(3):
            for kk in range(3):
                a_size = part_sizes[ii]
                b_size = part_sizes[jj]
                c_size = part_sizes[kk]

                i_idx = parts[ii]
                j_idx = parts[jj]
                k_idx = parts[kk]

                if a_size == 2 and b_size == 2 and c_size == 2:
                    i0, i1 = i_idx[0], i_idx[1]
                    j0, j1 = j_idx[0], j_idx[1]
                    k0, k1 = k_idx[0], k_idx[1]

                    a_entries = {
                        'a11': i0 * m + j0,
                        'a12': i0 * m + j1,
                        'a21': i1 * m + j0,
                        'a22': i1 * m + j1
                    }
                    b_entries = {
                        'b11': j0 * p + k0,
                        'b12': j0 * p + k1,
                        'b21': j1 * p + k0,
                        'b22': j1 * p + k1
                    }
                    c_entries = {
                        'C11': k0 * n + i0,
                        'C12': k1 * n + i0,
                        'C21': k0 * n + i1,
                        'C22': k1 * n + i1
                    }

                    strassen_defs = [
                        ({'a11': 1, 'a22': 1}, {'b11': 1, 'b22': 1}, {'C11': 1, 'C22': 1}),
                        ({'a21': 1, 'a22': 1}, {'b11': 1}, {'C21': 1}),
                        ({'a11': 1}, {'b12': 1, 'b22': -1}, {'C12': 1, 'C22': 1}),
                        ({'a22': 1}, {'b21': 1, 'b11': -1}, {'C11': 1, 'C21': 1}),
                        ({'a11': 1, 'a12': 1}, {'b22': 1}, {'C11': -1, 'C12': 1}),
                        ({'a21': 1, 'a11': -1}, {'b11': 1, 'b12': 1}, {'C22': 1}),
                        ({'a12': 1, 'a22': -1}, {'b21': 1, 'b22': 1}, {'C11': 1, 'C22': -1}),
                    ]

                    for u_coeff, v_coeff, w_coeff in strassen_defs:
                        for entry, coeff in u_coeff.items():
                            row = a_entries[entry]
                            U_rows[row][col] = U_rows[row].get(col, 0) + coeff
                        for entry, coeff in v_coeff.items():
                            row = b_entries[entry]
                            V_rows[row][col] = V_rows[row].get(col, 0) + coeff
                        for entry, coeff in w_coeff.items():
                            row = c_entries[entry]
                            W_rows[row][col] = W_rows[row].get(col, 0) + coeff
                        col += 1

                else:
                    for ai in range(a_size):
                        for bj in range(b_size):
                            for ck in range(c_size):
                                i = i_idx[ai]
                                j = j_idx[bj]
                                k = k_idx[ck]
                                U_rows[i * m + j][col] = 1
                                V_rows[j * p + k][col] = 1
                                W_rows[k * n + i][col] = 1
                                col += 1

    rank = col

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(n * m):
        for c, val in U_rows[r].items():
            U[r][c] = val
    for r in range(m * p):
        for c, val in V_rows[r].items():
            V[r][c] = val
    for r in range(p * n):
        for c, val in W_rows[r].items():
            W[r][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
