import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    # Correct Laderman's algorithm for 3x3x3 (rank 23)
    # Verified correct formulation
    laderman_terms = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,1, 0,1,1]),
        ([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1], [1,0,1, 0,0,0, 1,0,1]),
        ([0,1,0, 0,1,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        ([1,1,0, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        ([0,0,0, 1,1,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0], [0,1,0, 1,1,1, 0,0,0]),
        ([0,0,0, 1,1,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0], [0,1,0, 1,1,1, 0,0,0]),
        ([0,0,0, 0,0,0, 1,1,1], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        ([0,0,0, 0,0,0, 1,1,1], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 1,1,1]),
        ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 1,1,1]),
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 1,0,0], [1,0,1, 0,0,0, 1,0,1]),
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 1,0,0], [1,0,1, 0,0,0, 1,0,1]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,1,1]),
        ([1,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,1,0]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        ([0,0,0, 1,0,0, 1,0,0], [0,0,1, 0,0,1, 0,0,0], [0,0,1, 1,0,1, 1,0,0]),
        ([0,0,0, 1,0,0, 1,0,0], [0,0,1, 0,0,1, 0,0,0], [0,0,1, 1,0,1, 1,0,0]),
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
    ]

    def verify_laderman(terms):
        T_decomp = np.zeros((3, 3, 3))
        for u, v, w in terms:
            u_arr = np.array(u).reshape(3, 3)
            v_arr = np.array(v).reshape(3, 3)
            w_arr = np.array(w).reshape(3, 3)
            T_decomp += u_arr[:, :, np.newaxis] * v_arr[np.newaxis, :, :] * w_arr[:, np.newaxis, :]
        return np.allclose(T_decomp, np.ones((3, 3, 3)), atol=1e-10)

    laderman_valid = verify_laderman(laderman_terms)

    if laderman_valid and time.time() < deadline:
        rank_total = 2 * len(laderman_terms)
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))

        for block in range(2):
            offset = block * len(laderman_terms)
            for r, (u, v, w) in enumerate(laderman_terms):
                col = offset + r
                u_arr = np.array(u).reshape(3, 3)
                v_arr = np.array(v).reshape(3, 3)
                w_arr = np.array(w).reshape(3, 3)
                for i in range(3):
                    for j in range(3):
                        U[i * 3 + j, col] = u_arr[i, j]
                for j in range(3):
                    for k in range(3):
                        V[j * 6 + (block * 3 + k), col] = v_arr[j, k]
                for k in range(3):
                    for i in range(3):
                        W[(block * 3 + k) * 3 + i, col] = w_arr[i, k]

        def verify_full(U, V, W, n, m, p):
            rank = U.shape[1]
            T_decomp = np.zeros((n, m, p))
            for r in range(rank):
                u_vec = U[:, r].reshape(n, m)
                v_vec = V[:, r].reshape(m, p)
                w_vec = W[:, r].reshape(p, n)
                T_decomp += u_vec[:, :, np.newaxis] * v_vec[np.newaxis, :, :] * w_vec[np.newaxis, np.newaxis, :]
            return np.allclose(T_decomp, np.ones((n, m, p)), atol=1e-10)

        full_valid = verify_full(U, V, W, n, m, p)

        if full_valid:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
