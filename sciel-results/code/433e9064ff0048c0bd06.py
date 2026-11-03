import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # U2 row index = i*2+j (A entry), V2 row index = j*2+k (B entry), W2 row index = k*2+i (C entry).
    # The decomposition satisfies: sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] = 1 for all i,j,k in {0,1}.
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

    # Build 4x4 via Kronecker product of the 2x2 Strassen factors.
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # The rows of np.kron(A2, A2) are indexed by (a, b) where the row is a*4 + b,
    # with a being the row index in the first factor and b in the second.
    # For the 4x4 matrix with row index i*4+j, we decompose:
    #   i = i_hi * 2 + i_lo,  j = j_hi * 2 + j_lo
    # The Kronecker row index is:
    #   a = i_hi * 2 + j_hi  (row in first 2x2 block, treating i_hi as row, j_hi as col)
    #   b = i_lo * 2 + j_lo  (row in second 2x2 block)
    #   R = a * 4 + b
    # We need a permutation such that fixed_row[S] = kron_row[R] where S = i*4+j.
    perm = np.empty(16, dtype=int)
    for i in range(4):
        for j in range(4):
            S = i * 4 + j
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            a = i_hi * 2 + j_hi
            b = i_lo * 2 + j_lo
            R = a * 4 + b
            perm[S] = R

    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]

    # Verify 4x4 correctness
    ok4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.sum(U4_fixed[i*4+j, :] * V4_fixed[j*4+k, :] * W4_fixed[k*4+i, :]))
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

    if time.time() - start_time > deadline:
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
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
