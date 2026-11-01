import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    num_blocks = p // 2
    rank = num_blocks * 7

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]

            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val

            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b, col] = val

            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w, col] = val

    # Verify
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We check for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Vectorized dot product
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)

    if max_err > 1e-10:
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
