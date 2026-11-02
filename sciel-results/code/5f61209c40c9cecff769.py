import numpy as np
import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138

    n, m, p = 2, 2, 13

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Determine optimal partition of p columns into blocks of size 1 (rank 4) and 2 (rank 7)
        # Minimize 4*n1 + 7*n2 subject to n1 + 2*n2 = p
        # n1 = p - 2*n2
        # Cost = 4*(p - 2*n2) + 7*n2 = 4p - 8*n2 + 7*n2 = 4p - n2
        # Maximize n2 such that n1 >= 0 => 2*n2 <= p => n2 <= p//2
        n2_max = p // 2
        n2_opt = n2_max
        n1_opt = p - 2 * n2_opt
        
        rank_opt = 4 * n1_opt + 7 * n2_opt
        
        U = np.zeros((n * m, rank_opt))
        V = np.zeros((m * p, rank_opt))
        W = np.zeros((p * n, rank_opt))

        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]

        col = 0
        j_current = 0
        
        # Place n2_opt Strassen blocks
        for _ in range(n2_opt):
            j0 = j_current
            j1 = j_current + 1
            
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]

                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]

                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]

                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]

                col += 1
            j_current += 2

        # Place n1_opt Schoolbook blocks
        for _ in range(n1_opt):
            j_last = j_current
            for i in range(n):
                for k in range(m):
                    U[i * m + k, col] = 1.0
                    V[k * p + j_last, col] = 1.0
                    W[j_last * n + i, col] = 1.0
                    col += 1
            j_current += 1

        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0

        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))

        if error < 1e-10:
            best_rank = rank_opt
            best_U, best_V, best_W = U, V, W

    except Exception:
        pass

    if time.time() > deadline:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
