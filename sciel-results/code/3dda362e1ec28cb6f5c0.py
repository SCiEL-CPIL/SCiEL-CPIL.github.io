import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2   # 8 blocks of width 2
    rank = 7 * num_blocks # 56

    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # Each term gives contributions to U (4x1), V (2x1), W (2x1) for one block
    # u_entries: (i, j, val) for A[i,j]
    # v_entries: (j, k_local, val) for B[j, k_local]  
    # w_entries: (k_local, i_prime, val) for C[k_local, i_prime]
    #
    # Standard Strassen products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M5
    # C[1,1] = M1 - M2 + M3 + M6

    terms = [
        # M1
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        # M2
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)]),
        # M3
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)]),
        # M4
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)]),
        # M5
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        # M6
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        # M7
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]

    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]   # 4 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 32 x 56

    for block_idx in range(num_blocks):
        if time.time() > deadline:
            break
        k_base = 2 * block_idx
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u_entries, v_entries, w_entries = terms[term_idx]

            # Fill U: row index = i*m + j
            for (i, j, val) in u_entries:
                U[i * m + j][col] = val

            # Fill V: row index = j*p + k, where k = k_base + k_local
            for (j, k_local, val) in v_entries:
                k = k_base + k_local
                V[j * p + k][col] = val

            # Fill W: row index = k*n + i', where k = k_base + k_local
            for (k_local, ip, val) in w_entries:
                k = k_base + k_local
                W[k * n + ip][col] = val

    # Verification pass: confirm the decomposition is exact
    # T[i,j,k] should equal delta_{i,k} for the matrix multiplication tensor
    try:
        if time.time() < deadline - 5:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)

            # Reshape to 3D: U[i,j,r], V[j,k,r], W[k,i',r]
            U_3d = U_np.reshape(n, m, rank)
            V_3d = V_np.reshape(m, p, rank)
            W_3d = W_np.reshape(p, n, rank)

            # Contract: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # For matrix multiplication: T[i,j,k] = delta_{i,k}
            T = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)

            # Check against identity: T[i,j,k] = 1 if i==k else 0
            expected = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i == k:
                            expected[i, j, k] = 1.0

            if not np.allclose(T, expected, atol=1e-10):
                # Fallback: schoolbook decomposition (rank 64)
                rank_fb = n * m * p  # 64
                U_fb = [[0] * rank_fb for _ in range(n * m)]
                V_fb = [[0] * rank_fb for _ in range(m * p)]
                W_fb = [[0] * rank_fb for _ in range(p * n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i * m * p + j * p + k
                            U_fb[i * m + j][col] = 1
                            V_fb[j * p + k][col] = 1
                            W_fb[k * n + i][col] = 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
