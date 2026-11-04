import numpy as np

def construct():

    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # U2 row index = i*2+j, V2 row index = j*2+k, W2 row index = k*2+i.
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Quick sanity check for 2x2 Strassen correctness
    ok2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :]))
                if s != 1:
                    ok2 = False
                    break
            if not ok2:
                break
        if not ok2:
            break

    if not ok2:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build 4x4 factors from 2x2 Strassen factors
    # U4[i*4+j, c1*7+c2] = U2[i_hi*2+j_hi, c1] * U2[i_lo*2+j_lo, c2]
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            idx_u = i * 4 + j
            row_u_hi = U2[i_hi * 2 + j_hi, :]
            row_u_lo = U2[i_lo * 2 + j_lo, :]
            # Kronecker product of two rows
            U4[idx_u, :] = np.kron(row_u_hi, row_u_lo)

    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            idx_v = j * 4 + k
            row_v_hi = V2[j_hi * 2 + k_hi, :]
            row_v_lo = V2[j_lo * 2 + k_lo, :]
            V4[idx_v, :] = np.kron(row_v_hi, row_v_lo)

    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            idx_w = k * 4 + i
            row_w_hi = W2[k_hi * 2 + i_hi, :]
            row_w_lo = W2[k_lo * 2 + i_lo, :]
            W4[idx_w, :] = np.kron(row_w_hi, row_w_lo)

    # Verify 4x4 correctness
    ok4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :]))
                if s != 1:
                    ok4 = False
                    break
            if not ok4:
                break
        if not ok4:
            break

    if not ok4:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
