import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18

    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1

            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1

            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1

            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7

        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1

        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1

        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4

        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2

        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W):
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True

    def reduce_rank(U, V, W):
        """Remove linearly dependent columns from the factor matrices."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        r = U_arr.shape[1]

        # Check each column: if it's a linear combination of others, remove it
        keep = list(range(r))
        for c in range(r):
            if time.time() > deadline - 30:
                break
            if c not in keep:
                continue
            # Try to see if column c can be expressed as combination of others
            others = [i for i in keep if i != c]
            if not others:
                continue

            # Stack the columns from U, V, W for column c
            col_c = np.concatenate([U_arr[:, c], V_arr[:, c], W_arr[:, c]])
            col_others = np.concatenate([
                U_arr[:, others],
                V_arr[:, others],
                W_arr[:, others]
            ], axis=1)

            # Check if col_c is in the span of col_others
            try:
                rank_others = np.linalg.matrix_rank(col_others)
                rank_with = np.linalg.matrix_rank(np.column_stack([col_others, col_c]))
                if rank_with == rank_others:
                    keep.remove(c)
            except:
                pass

        if len(keep) == r:
            return None

        U_new = np.array(U, dtype=np.int64)[:, keep].tolist()
        V_new = np.array(V, dtype=np.int64)[:, keep].tolist()
        W_new = np.array(W, dtype=np.int64)[:, keep].tolist()

        # Verify
        if verify_fast(U_new, V_new, W_new):
            return {"U": U_new, "V": V_new, "W": W_new}
        return None

    def try_rank_structured(r, max_trials=20):
        """Try to find a rank-r decomposition using structured sparse patterns."""
        if time.time() > deadline - 20:
            return None

        for trial in range(max_trials):
            if time.time() > deadline - 15:
                return None

            # Initialize with sparse random matrices
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)

            # Fill V and W with sparse random entries
            for c in range(r):
                # V column: ~3 non-zeros
                v_indices = np.random.choice(m_rows, size=3, replace=False)
                for idx in v_indices:
                    V[idx, c] = np.random.choice([-1, 1, -2, 2])

                # W column: ~2 non-zeros
                w_indices = np.random.choice(p_rows, size=2, replace=False)
                for idx in w_indices:
                    W[idx, c] = np.random.choice([-1, 1, -2, 2])

            U = np.zeros((n_rows, r), dtype=np.int64)

            valid = True
            for i in range(n):
                for j in range(m):
                    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]

                    # Solve M @ u = 1 (least squares)
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        valid = False
                        break

                    u_int = np.round(u).astype(np.int64)

                    # Check exactness
                    if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                        valid = False
                        break

                    U[i*m+j, :] = u_int

            if not valid:
                continue

            # Full verification
            if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                reduced = reduce_rank(result["U"], result["V"], result["W"])
                if reduced:
                    return reduced
                return result

        return None

    # Try to find a lower rank solution
    best_result = None
    best_rank = 50

    # First, try to reduce the known rank-50 solution
    base_sol = current_solution()
    reduced = reduce_rank(base_sol["U"], base_sol["V"], base_sol["W"])
    if reduced:
        best_result = reduced
        best_rank = len(reduced["U"][0])

    # Try structured random search for lower ranks
    for r in range(best_rank - 1, max(best_rank - 10, 35), -1):
        if time.time() > deadline - 25:
            break

        res = try_rank_structured(r, max_trials=15)
        if res:
            best_result = res
            best_rank = len(res["U"][0])
            break

    if best_result:
        return best_result

    # Fallback to rank 50
    return current_solution()
