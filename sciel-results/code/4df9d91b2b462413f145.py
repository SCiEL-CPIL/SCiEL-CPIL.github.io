import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 decomposition (rank 7)
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=float)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)

    # Part 1: 4x4x4 decomposition with rank 49 (tensor product of two Strassen)
    rank1 = 49
    U1 = np.zeros((n*m, rank1), dtype=float)
    V1 = np.zeros((m*p, rank1), dtype=float)
    W1 = np.zeros((p*n, rank1), dtype=float)

    # Build 4x4x4 via tensor product of two 2x2x2 Strassen
    # For i,j,k in 0..3, decompose into (i1,i2),(j1,j2),(k1,k2) where i=2*i1+i2 etc.
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            row_U = i * m + j
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    U1[row_U, col] = S_U[i1*2+j1, c1] * S_U[i2*2+j2, c2]

    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            row_V = j * p + k
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    V1[row_V, col] = S_V[j1*2+k1, c1] * S_V[j2*2+k2, c2]

    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            row_W = k * n + i
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    W1[row_W, col] = S_W[k1*2+i1, c1] * S_W[k2*2+i2, c2]

    # Part 2: 4x4x2 decomposition with rank 28 (tiling)
    rank2 = 28
    U2 = np.zeros((n*m, rank2), dtype=float)
    V2 = np.zeros((m*p, rank2), dtype=float)
    W2 = np.zeros((p*n, rank2), dtype=float)

    # Tile the 4x4x2 as four 2x2x2 blocks (bi in {0,1} for rows, bj in {0,1} for cols)
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = (bi * 2 + bj) * 7 + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U2[i * m + j, col] = S_U[di * 2 + dj, c]

                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = 4 + dk
                        V2[j * p + k, col] = S_V[dj * 2 + dk, c]

                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        W2[k * n + i, col] = S_W[dk * 2 + di, c]

    # Combine
    total_rank = rank1 + rank2
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])

    # Verify correctness
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(total_rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break

    if not verified:
        # Fallback: naive decomposition (rank = n*m*p = 96)
        U_naive = np.zeros((n*m, n*m*p))
        V_naive = np.zeros((m*p, n*m*p))
        W_naive = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, idx] = 1.0
                    V_naive[j*p+k, idx] = 1.0
                    W_naive[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    # Try to eliminate zero columns
    col_sums = np.sum(U**2, axis=0) + np.sum(V**2, axis=0) + np.sum(W**2, axis=0)
    keep_cols = [i for i in range(total_rank) if col_sums[i] > 1e-15]
    
    if len(keep_cols) < total_rank:
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]

    final_rank = U.shape[1]
    
    # Convert to lists with integer/half-integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure values are clean (round small values to 0, etc.)
    def clean_list(lst):
        cleaned = []
        for row in lst:
            clean_row = []
            for v in row:
                v_rounded = round(v, 10)
                if abs(v_rounded) < 1e-10:
                    v_rounded = 0.0
                clean_row.append(v_rounded)
            cleaned.append(clean_row)
        return cleaned
    
    U_final = clean_list(U_list)
    V_final = clean_list(V_list)
    W_final = clean_list(W_list)
    
    return {"U": U_final, "V": V_final, "W": W_final}
