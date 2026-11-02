import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7

    # Schoolbook decomposition as fallback (rank 56)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    # Deterministic block decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for 2x2x1 blocks
    # Total rank: 50

    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

    # Build the decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    # Verify the decomposition
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Efficient verification using matrix multiplication
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We check T[i, j, k] == 1 if j == jp (wait, T is for fixed i, k? No)
        # T_{i,j,k} is 1 if the indices match the identity?
        # The tensor T has T[i, j, k] = 1 for all i, j, k? No.
        # T is the matrix multiplication tensor.
        # T_{i, j, k} = 1 if the operation C_{i,k} = sum_j A_{i,j} B_{j,k} is correct.
        # The decomposition is sum_c U_{i,j}^c V_{j,k}^c W_{i,k}^c = delta_{j,j'}? No.
        # The identity is: sum_c U_{i,j}^c V_{j,k}^c W_{i,k}^c = 1 for all i,j,k?
        # No, the tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1.
        # So we check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if col == rank and verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    else:
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
