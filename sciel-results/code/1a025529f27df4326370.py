"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1

        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1

        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1

        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    for j_block in range(2):
        for k_block in range(3):
            add_strassen_block(0, j_block*2, k_block*2, (j_block*3 + k_block)*7)

    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    def verify(U_c, V_c, W_c, tol=1e-9):
        U_a = np.array(U_c, dtype=np.float64)
        V_a = np.array(V_c, dtype=np.float64)
        W_a = np.array(W_c, dtype=np.float64)
        cur_rank = U_a.shape[1]
        M = np.zeros((n*m*p, cur_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = (i*m + j) * p + k
                    for r in range(cur_rank):
                        M[row_idx, r] = U_a[i*m+j, r] * V_a[j*p+k, r] * W_a[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = (i*m + j) * p + k
                    if abs(M[row_idx].sum() - 1.0) > tol:
                        return False
        return True

    def try_reduce(U_c, V_c, W_c, max_iters=30):
        cur_U, cur_V, cur_W = [row[:] for row in U_c], [row[:] for row in V_c], [row[:] for row in W_c]
        cur_rank = len(cur_U[0])

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            U_a = np.array(cur_U, dtype=np.float64)
            V_a = np.array(cur_V, dtype=np.float64)
            W_a = np.array(cur_W, dtype=np.float64)

            M = np.zeros((n*m*p, cur_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = (i*m + j) * p + k
                        for r in range(cur_rank):
                            M[row_idx, r] = U_a[i*m+j, r] * V_a[j*p+k, r] * W_a[k*n+i, r]

            try:
                s_vals = np.linalg.svd(M, compute_uv=False)
            except np.linalg.LinAlgError:
                break

            tol = 1e-10
            eff_rank = np.sum(s_vals > tol)

            if eff_rank >= cur_rank:
                break

            _, _, Vt = np.linalg.svd(M, full_matrices=False)
            null_vec = Vt[-1]

            drop_idx = -1
            for r in range(cur_rank):
                if abs(null_vec[r]) < 1e-10:
                    drop_idx = r
                    break

            if drop_idx == -1:
                drop_idx = int(np.argmin(np.abs(null_vec)))
                c_s = -null_vec / null_vec[drop_idx]

                new_U = []
                new_V = []
                new_W = []
                for s in range(cur_rank):
                    if s == drop_idx:
                        continue
                    scale = 1.0 + c_s[s]
                    new_U.append([U_a[i, s] * scale for i in range(n*m)])
                    new_V.append([V_a[i, s] for i in range(m*p)])
                    new_W.append([W_a[i, s] for i in range(p*n)])

                if verify(new_U, new_V, new_W):
                    cur_U, cur_V, cur_W = new_U, new_V, new_W
                    cur_rank -= 1
                else:
                    break
            else:
                cur_U = [row[:drop_idx] + row[drop_idx+1:] for row in cur_U]
                cur_V = [row[:drop_idx] + row[drop_idx+1:] for row in cur_V]
                cur_W = [row[:drop_idx] + row[drop_idx+1:] for row in cur_W]
                cur_rank -= 1

        return cur_U, cur_V, cur_W, cur_rank

    U_r, V_r, W_r, rank_r = try_reduce(U, V, W, max_iters=20)

    if rank_r < rank:
        return {"U": U_r, "V": V_r, "W": W_r}

    best_U, best_V, best_W, best_rank = U, V, W, rank

    for trial in range(50):
        if time.time() > deadline - 20:
            break
        if best_rank < 2:
            break

        r1 = random.randint(0, best_rank - 1)
        r2 = random.randint(0, best_rank - 1)
        if r1 == r2:
            continue

        alpha = random.choice([0.5, -0.5, 1.0, -1.0, 0.25, -0.25])

        U_a = np.array(best_U, dtype=np.float64)
        V_a = np.array(best_V, dtype=np.float64)
        W_a = np.array(best_W, dtype=np.float64)

        new_U = []
        new_V = []
        new_W = []
        for s in range(best_rank):
            if s == r2:
                continue
            if s == r1:
                new_U.append(list(U_a[:, s] * (1 + alpha)))
                new_V.append(list(V_a[:, s]))
                new_W.append(list(W_a[:, s]))
            else:
                new_U.append(list(U_a[:, s]))
                new_V.append(list(V_a[:, s]))
                new_W.append(list(W_a[:, s]))

        new_rank = best_rank - 1
        if verify(new_U, new_V, new_W, tol=1e-6):
            best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
            break

    if best_rank < rank:
        return {"U": best_U, "V": best_W, "W": best_W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
