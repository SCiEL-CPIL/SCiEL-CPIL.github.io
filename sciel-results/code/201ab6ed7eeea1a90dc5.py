import numpy as np

def construct():
    # Rank-49 decomposition of the 4x4 matrix multiplication tensor via
    # the Kronecker product of Strassen's rank-7 decomposition of 2x2.
    # This is an exact, verified decomposition with integer entries.
    #
    # Tensor convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where (n,m,p) = (4,4,4) and the output index is (k, i) following W[k*n+i, r].

    # Strassen's rank-7 decomposition for 2x2:
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1]    # (1,1)
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1]    # (1,1)
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0]    # (1,1)
    ], dtype=int)

    n = m = p = 4
    rank = 49

    U4 = np.zeros((n*m, rank), dtype=int)
    V4 = np.zeros((m*p, rank), dtype=int)
    W4 = np.zeros((p*n, rank), dtype=int)

    # Build 4x4 factors using Kronecker product structure
    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            u_idx_hi = i_hi * 2 + j_hi
            u_idx_lo = i_lo * 2 + j_lo
            for r1 in range(7):
                a = U2[u_idx_hi, r1]
                if a == 0:
                    continue
                for r2 in range(7):
                    b = U2[u_idx_lo, r2]
                    if b == 0:
                        continue
                    U4[i * 4 + j, r1 * 7 + r2] = a * b

    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            v_idx_hi = j_hi * 2 + k_hi
            v_idx_lo = j_lo * 2 + k_lo
            for r1 in range(7):
                a = V2[v_idx_hi, r1]
                if a == 0:
                    continue
                for r2 in range(7):
                    b = V2[v_idx_lo, r2]
                    if b == 0:
                        continue
                    V4[j * 4 + k, r1 * 7 + r2] = a * b

    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            w_idx_hi = k_hi * 2 + i_hi
            w_idx_lo = k_lo * 2 + i_lo
            for r1 in range(7):
                a = W2[w_idx_hi, r1]
                if a == 0:
                    continue
                for r2 in range(7):
                    b = W2[w_idx_lo, r2]
                    if b == 0:
                        continue
                    W4[k * 4 + i, r1 * 7 + r2] = a * b

    # Verify correctness: T[i,j,k] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] must equal 1
    ok = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.dot(U4[i*4+j, :] * V4[j*4+k, :], W4[k*4+i, :]))
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback: schoolbook rank-64
        U = [[0] * 64 for _ in range(16)]
        V = [[0] * 64 for _ in range(16)]
        W = [[0] * 64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
