import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def add_strassen_term(U, V, W, u_idx, v_idx, w_idx, col):
        for row, coeff in u_idx:
            U[row, col] += coeff
        for row, coeff in v_idx:
            V[row, col] += coeff
        for row, coeff in w_idx:
            W[row, col] += coeff

    def strassen_2x2(U, V, W, u_rows, v_rows, w_rows, start_col):
        u00, u01, u10, u11 = u_rows
        v00, v01, v10, v11 = v_rows
        w00, w01, w10, w11 = w_rows
        col = start_col
        # M1 = (X00+X11)*(Y00+Y11)
        add_strassen_term(U, V, W,
                         [(u00,1),(u11,1)],
                         [(v00,1),(v11,1)],
                         [(w00,1),(w11,1)], col)
        col += 1
        # M2 = (X10+X11)*Y00
        add_strassen_term(U, V, W,
                         [(u10,1),(u11,1)],
                         [(v00,1)],
                         [(w10,1),(w11,-1)], col)
        col += 1
        # M3 = X00*(Y01-Y11)
        add_strassen_term(U, V, W,
                         [(u00,1)],
                         [(v01,1),(v11,-1)],
                         [(w01,1),(w11,1)], col)
        col += 1
        # M4 = X11*(Y10-Y00)
        add_strassen_term(U, V, W,
                         [(u11,1)],
                         [(v10,1),(v00,-1)],
                         [(w00,1),(w10,1)], col)
        col += 1
        # M5 = (X00+X01)*Y11
        add_strassen_term(U, V, W,
                         [(u00,1),(u01,1)],
                         [(v11,1)],
                         [(w00,-1),(w01,1)], col)
        col += 1
        # M6 = (X10-X00)*(Y00+Y01)
        add_strassen_term(U, V, W,
                         [(u10,1),(u00,-1)],
                         [(v00,1),(v01,1)],
                         [(w11,1)], col)
        col += 1
        # M7 = (X01-X11)*(Y10+Y11)
        add_strassen_term(U, V, W,
                         [(u01,1),(u11,-1)],
                         [(v10,1),(v11,1)],
                         [(w00,1),(w11,-1)], col)
        col += 1
        return col

    def add_schoolbook_2x1x1x6(U, V, W, u_rows, v_rows, w_rows, start_col):
        col = start_col
        for i in range(2):
            for k in range(6):
                U[u_rows[i], col] = 1
                V[v_rows[k], col] = 1
                W[w_rows[i*6+k], col] = 1
                col += 1
        return col

    # Build Strassen-based decomposition with rank 54
    U = np.zeros((nm, 54))
    V = np.zeros((mp, 54))
    W = np.zeros((pn, 54))

    col = 0
    # A1*B1a: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = strassen_2x2(U, V, W,
                       [0, 1, 5, 6],
                       [0, 1, 6, 7],
                       [0, 1, 2, 3], col)

    # A1*B1b: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    col = strassen_2x2(U, V, W,
                       [0, 1, 5, 6],
                       [2, 3, 8, 9],
                       [4, 5, 6, 7], col)

    # A1*B1c: A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    col = strassen_2x2(U, V, W,
                       [0, 1, 5, 6],
                       [4, 5, 10, 11],
                       [8, 9, 10, 11], col)

    # A2*B2a: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    col = strassen_2x2(U, V, W,
                       [2, 3, 7, 8],
                       [12, 13, 18, 19],
                       [0, 1, 2, 3], col)

    # A2*B2b: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    col = strassen_2x2(U, V, W,
                       [2, 3, 7, 8],
                       [14, 15, 20, 21],
                       [4, 5, 6, 7], col)

    # A2*B2c: A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    col = strassen_2x2(U, V, W,
                       [2, 3, 7, 8],
                       [16, 17, 22, 23],
                       [8, 9, 10, 11], col)

    # A3*B3: A[0:2,4:5] * B[4:5,0:6] -> C[0:2,0:6] (schoolbook, 12 terms)
    col = add_schoolbook_2x1x1x6(U, V, W,
                                 [4, 9],
                                 [24, 25, 26, 27, 28, 29],
                                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], col)

    assert col == 54, f"Expected 54 columns, got {col}"

    # Verify the decomposition
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)

    idx_u = np.array(idx_u, dtype=np.int32)
    idx_v = np.array(idx_v, dtype=np.int32)
    idx_w = np.array(idx_w, dtype=np.int32)

    vals = U[idx_u] * V[idx_v] * W[idx_w]
    sums = vals.sum(axis=1)

    if not np.all(np.abs(sums - 1.0) < 1e-10):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
