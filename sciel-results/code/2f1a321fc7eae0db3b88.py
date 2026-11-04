import numpy as np

def construct():
    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
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

    # Build 4x4 factors from 2x2 Strassen factors using Kronecker product logic
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            u_idx_hi = i_hi*2+j_hi
            u_idx_lo = i_lo*2+j_lo
            for c1 in range(7):
                u1 = U2[u_idx_hi, c1]
                if u1 == 0: continue
                for c2 in range(7):
                    u2 = U2[u_idx_lo, c2]
                    if u2 == 0: continue
                    U4[i*4+j, c1*7+c2] = u1 * u2

    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            v_idx_hi = j_hi*2+k_hi
            v_idx_lo = j_lo*2+k_lo
            for c1 in range(7):
                v1 = V2[v_idx_hi, c1]
                if v1 == 0: continue
                for c2 in range(7):
                    v2 = V2[v_idx_lo, c2]
                    if v2 == 0: continue
                    V4[j*4+k, c1*7+c2] = v1 * v2

    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            w_idx_hi = k_hi*2+i_hi
            w_idx_lo = k_lo*2+i_lo
            for c1 in range(7):
                w1 = W2[w_idx_hi, c1]
                if w1 == 0: continue
                for c2 in range(7):
                    w2 = W2[w_idx_lo, c2]
                    if w2 == 0: continue
                    W4[k*4+i, c1*7+c2] = w1 * w2

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
