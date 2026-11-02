import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 145

    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}

    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k_idx = bm * n + ai
                T[i, j, k_idx] = 1.0

    # Verify current decomposition
    T_decomp = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
    residual = np.linalg.norm(T_decomp - T)

    if residual > 1e-10:
        return {"U": U, "V": V, "W": W}

    # Build tensor matrix for SVD
    nm, mp, pn = n*m, m*p, p*n
    total_size = nm * mp * pn

    mat = np.zeros((total_size, rank))
    for c in range(rank):
        mat[:, c] = np.kron(np.kron(U_np[:, c], V_np[:, c]), W_np[:, c])

    if time.time() > deadline - 40:
        return {"U": U, "V": V, "W": W}

    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()

    # SVD to find null space
    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        tol = max(1e-12, s[0] * max(nm, mp, pn) * 1e-10)
        num_rank = int(np.sum(s > tol))

        if num_rank < rank:
            null_basis = Vt[num_rank:]  # shape (rank - num_rank, rank)
            n_null = null_basis.shape[0]

            # Strategy 1: Greedy elimination using null space vectors
            eliminated = set()
            for nv in null_basis:
                if time.time() > deadline - 25:
                    break
                max_idx = -1
                max_val = 0
                for c in range(rank):
                    if c not in eliminated:
                        if abs(nv[c]) > max_val:
                            max_val = abs(nv[c])
                            max_idx = c
                if max_idx >= 0 and max_val > tol:
                    eliminated.add(max_idx)

            if eliminated:
                keep_cols = [c for c in range(rank) if c not in eliminated]
                U_new = U_np[:, keep_cols]
                V_new = V_np[:, keep_cols]
                W_new = W_np[:, keep_cols]

                T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                res = np.linalg.norm(T_new - T)

                if res < 1e-10:
                    best_rank = len(keep_cols)
                    best_U = U_new.copy()
                    best_V = V_new.copy()
                    best_W = W_new.copy()

            # Strategy 2: Try removing each single column
            for drop_col in range(rank):
                if time.time() > deadline - 15:
                    break
                U_new = np.delete(U_np, drop_col, axis=1)
                V_new = np.delete(V_np, drop_col, axis=1)
                W_new = np.delete(W_np, drop_col, axis=1)

                T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                res = np.linalg.norm(T_new - T)

                if res < 1e-10:
                    new_rank = U_new.shape[1]
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U = U_new.copy()
                        best_V = V_new.copy()
                        best_W = W_new.copy()

            # Strategy 3: Try removing pairs of columns (only if time permits)
            if best_rank < rank and time.time() < deadline - 30:
                # Re-run SVD on the reduced set to find more dependencies
                U_cur = best_U
                V_cur = best_V
                W_cur = best_W
                r_cur = best_rank

                mat2 = np.zeros((total_size, r_cur))
                for c in range(r_cur):
                    mat2[:, c] = np.kron(np.kron(U_cur[:, c], V_cur[:, c]), W_cur[:, c])

                try:
                    _, s2, Vt2 = np.linalg.svd(mat2, full_matrices=False)
                    tol2 = max(1e-12, s2[0] * max(nm, mp, pn) * 1e-10)
                    num_rank2 = int(np.sum(s2 > tol2))

                    if num_rank2 < r_cur:
                        null_basis2 = Vt2[num_rank2:]
                        eliminated2 = set()
                        for nv in null_basis2:
                            if time.time() > deadline - 15:
                                break
                            max_idx = -1
                            max_val = 0
                            for c in range(r_cur):
                                if c not in eliminated2:
                                    if abs(nv[c]) > max_val:
                                        max_val = abs(nv[c])
                                        max_idx = c
                            if max_idx >= 0 and max_val > tol2:
                                eliminated2.add(max_idx)

                        if eliminated2:
                            keep_cols2 = [c for c in range(r_cur) if c not in eliminated2]
                            U_new = U_cur[:, keep_cols2]
                            V_new = V_cur[:, keep_cols2]
                            W_new = W_cur[:, keep_cols2]

                            T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                            res = np.linalg.norm(T_new - T)

                            if res < 1e-10:
                                new_rank = U_new.shape[1]
                                if new_rank < best_rank:
                                    best_rank = new_rank
                                    best_U = U_new.copy()
                                    best_V = V_new.copy()
                                    best_W = W_new.copy()
                except Exception:
                    pass

            # Strategy 4: Try all single-column removals on the current best
            if best_rank < rank and time.time() < deadline - 10:
                U_cur = best_U
                V_cur = best_V
                W_cur = best_W
                r_cur = best_rank

                for drop_col in range(r_cur):
                    if time.time() > deadline - 8:
                        break
                    U_new = np.delete(U_cur, drop_col, axis=1)
                    V_new = np.delete(V_cur, drop_col, axis=1)
                    W_new = np.delete(W_cur, drop_col, axis=1)

                    T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                    res = np.linalg.norm(T_new - T)

                    if res < 1e-10:
                        new_rank = U_new.shape[1]
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_U = U_new.copy()
                            best_V = V_new.copy()
                            best_W = W_new.copy()
                            break  # restart with new best

    except Exception:
        pass

    if best_rank < rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    return {"U": U, "V": V, "W": W}
