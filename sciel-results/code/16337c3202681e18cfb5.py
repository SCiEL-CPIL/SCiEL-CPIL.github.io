import numpy as np
import time
from itertools import combinations

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

    def build_tensor_matrix(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        mat = np.zeros((total_size, r))
        for c in range(r):
            mat[:, c] = np.kron(np.kron(U_mat[:, c], V_mat[:, c]), W_mat[:, c])
        return mat

    def verify_decomp(U_mat, V_mat, W_mat):
        T_new = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
        return np.linalg.norm(T_new - T) < 1e-10

    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = rank

    # Strategy 1: Iterative single-column removal
    # Try removing each column one at a time, greedily
    U_cur = best_U.copy()
    V_cur = best_V.copy()
    W_cur = best_W.copy()
    r_cur = best_rank

    # Precompute tensor matrix for efficiency
    mat = build_tensor_matrix(U_cur, V_cur, W_cur)

    removed_cols = set()
    for iteration in range(10):  # up to 10 removals
        if time.time() > deadline - 30:
            break
        if r_cur <= 40:  # stop if we get to 40 or below
            break

        best_drop = -1
        best_drop_mat = None

        # Try each remaining column
        for c in range(r_cur):
            if c in removed_cols:
                continue
            if time.time() > deadline - 25:
                break

            # Remove column c
            keep = [i for i in range(r_cur) if i != c]
            U_new = U_cur[:, keep]
            V_new = V_cur[:, keep]
            W_new = W_cur[:, keep]

            if verify_decomp(U_new, V_new, W_new):
                best_drop = c
                best_drop_mat = (U_new, V_new, W_new)
                break  # Take the first valid removal

        if best_drop >= 0:
            U_cur, V_cur, W_cur = best_drop_mat
            r_cur -= 1
            removed_cols.add(best_drop)
            if r_cur < best_rank:
                best_rank = r_cur
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
        else:
            break

    # Strategy 2: Use SVD null space to find dependencies
    if time.time() < deadline - 20:
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        r_cur = best_rank

        mat = build_tensor_matrix(U_cur, V_cur, W_cur)

        try:
            _, s, Vt = np.linalg.svd(mat, full_matrices=False)
            tol = max(1e-12, s[0] * max(nm, mp, pn) * 1e-10)
            num_rank = int(np.sum(s > tol))

            if num_rank < r_cur:
                null_basis = Vt[num_rank:]  # shape (r_cur - num_rank, r_cur)
                n_null = null_basis.shape[0]

                # Use null space vectors to identify removable columns
                # A column c can be removed if there's a null vector with large weight on c
                # and we can express it in terms of others
                eliminated = set()
                for nv in null_basis:
                    if time.time() > deadline - 15:
                        break
                    # Find the column with maximum absolute value in this null vector
                    max_idx = -1
                    max_val = 0
                    for c in range(r_cur):
                        if c not in eliminated:
                            if abs(nv[c]) > max_val:
                                max_val = abs(nv[c])
                                max_idx = c
                    if max_idx >= 0 and max_val > tol:
                        # Check if removing this column still gives valid decomposition
                        keep = [i for i in range(r_cur) if i != max_idx]
                        U_new = U_cur[:, keep]
                        V_new = V_cur[:, keep]
                        W_new = W_cur[:, keep]
                        if verify_decomp(U_new, V_new, W_new):
                            eliminated.add(max_idx)
                            U_cur = U_new
                            V_cur = V_new
                            W_cur = W_new
                            r_cur -= 1
                            if r_cur < best_rank:
                                best_rank = r_cur
                                best_U = U_cur.copy()
                                best_V = V_cur.copy()
                                best_W = W_cur.copy()
        except Exception:
            pass

    # Strategy 3: Try pairs of column removals
    if best_rank > 40 and time.time() < deadline - 10:
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        r_cur = best_rank

        # Try removing pairs
        for i in range(r_cur):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, r_cur):
                if time.time() > deadline - 5:
                    break
                keep = [k for k in range(r_cur) if k != i and k != j]
                U_new = U_cur[:, keep]
                V_new = V_cur[:, keep]
                W_new = W_cur[:, keep]
                if verify_decomp(U_new, V_new, W_new):
                    if r_cur - 2 < best_rank:
                        best_rank = r_cur - 2
                        best_U = U_new.copy()
                        best_V = V_new.copy()
                        best_W = W_new.copy()
                    break
            if r_cur - 2 < best_rank:
                break

    # Final: return the best found
    if best_rank < rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    return {"U": U, "V": V, "W": W}
