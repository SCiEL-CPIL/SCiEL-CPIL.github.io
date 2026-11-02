import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 140  # safety margin before 150s kill

    # ---- Build the rank-50 decomposition (Strassen 2x2 blocks + schoolbook tail) ----
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

    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}

    # ---- Build the target tensor T[i,j,k] where i=(a,am), j=(bm,b), k=(k_idx=n*row+col) ----
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k_idx = bm * n + ai
                T[i, j, k_idx] = 1.0

    def verify(Um, Vm, Wm):
        Td = np.einsum('ic,jc,kc->ijk', Um, Vm, Wm)
        return np.linalg.norm(Td - T)

    # verify current
    if verify(U_np, V_np, W_np) > 1e-9:
        return {"U": U, "V": V, "W": W}

    # ---- Linear-dependency detection via SVD on the flattened rank-1 tensor matrix ----
    nm, mp, pn = n * m, m * p, p * n
    total_size = nm * mp * pn
    r = U_np.shape[1]
    mat = np.zeros((total_size, r))
    for c in range(r):
        mat[:, c] = np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c]).ravel()

    if time.time() > deadline - 20:
        return {"U": U, "V": V, "W": W}

    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        tol = s[0] * max(nm, mp, pn) * 1e-10
        num_rank = int(np.sum(s > tol))
    except Exception:
        num_rank = r

    best_U, best_V, best_W = U_np, V_np, W_np
    best_rank = r

    # ---- Eliminate redundant (linearly dependent) columns using projection test ----
    # A column c is redundant iff its flattened tensor is in the span of the others.
    # We iteratively find and drop such columns.
    if num_rank < r and time.time() < deadline - 15:
        keep = list(range(r))
        # Build the flattened matrix for the current set
        def build_mat(cols):
            m2 = np.zeros((total_size, len(cols)))
            for idx, c in enumerate(cols):
                m2[:, idx] = np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c]).ravel()
            return m2

        cols = list(range(r))
        M = build_mat(cols)
        # iterate: find a dependent column
        for _pass in range(r):
            if time.time() > deadline - 15:
                break
            if len(cols) <= num_rank:
                break
            # SVD of current M
            try:
                _, s2, _ = np.linalg.svd(M, full_matrices=False)
            except Exception:
                break
            tol2 = s2[0] * max(nm, mp, pn) * 1e-9
            cur_rank = int(np.sum(s2 > tol2))
            if cur_rank >= len(cols):
                break
            # find a column that is in the span of the others:
            # for each column c, check if it is in span of M without c
            dropped = False
            for idx in range(len(cols)):
                if time.time() > deadline - 15:
                    break
                # residual of column idx against span of the rest
                others = [j for j in range(len(cols)) if j != idx]
                if len(others) == 0:
                    continue
                Mrest = M[:, others]
                # project
                try:
                    # least squares: Mrest @ x = M[:, idx]
                    x, res, rk, sv = np.linalg.lstsq(Mrest, M[:, idx], rcond=None)
                    resid = np.linalg.norm(M[:, idx] - Mrest @ x)
                    scale = np.linalg.norm(M[:, idx]) + 1e-12
                    if resid / scale < 1e-7:
                        # column idx is redundant
                        cols = [c for j, c in enumerate(cols) if j != idx]
                        M = M[:, [j for j in range(len(M.shape[1])) if j != idx]]
                        dropped = True
                        break
                except Exception:
                    pass
            if not dropped:
                break

        if len(cols) < r:
            best_U = U_np[:, cols]
            best_V = V_np[:, cols]
            best_W = W_np[:, cols]
            best_rank = len(cols)
            # verify
            if verify(best_U, best_V, best_W) < 1e-9:
                pass
            else:
                # revert
                best_U, best_V, best_W = U_np, V_np, W_np
                best_rank = r

    if best_rank < r:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # ---- Polynomial-interpolation search for genuinely lower-rank decomposition ----
    # If a rank-r decomposition exists, then for random scalars t, the "evaluation"
    # U(t)=sum_c U_c t^c etc. yields a valid decomposition of T(t)=sum T_c t^c.
    # We test candidate ranks by checking consistency of a low-rank ansatz.
    # This is expensive; only attempt if time remains.
    if time.time() < deadline - 35:
        try:
            # Try to find a lower-rank decomposition via random sparse ansatz + verification
            # Use the known structure: the 2x3x9 tensor has a specific sparsity.
            # We attempt ranks from 44 up to 49.
            np.random.seed(12345)
            found = False
            for target_rank in range(44, min(49, r)):
                if time.time() > deadline - 25:
                    break
                # Random sparse initialization with small integer entries
                for attempt in range(60):
                    if time.time() > deadline - 25:
                        break
                    U_try = np.zeros((nm, target_rank))
                    V_try = np.zeros((mp, target_rank))
                    W_try = np.zeros((pn, target_rank))
                    for c in range(target_rank):
                        for rr in range(nm):
                            if np.random.random() < 0.25:
                                U_try[rr, c] = np.random.choice([-1, 1])
                        for rr in range(mp):
                            if np.random.random() < 0.2:
                                V_try[rr, c] = np.random.choice([-1, 1])
                        for rr in range(pn):
                            if np.random.random() < 0.2:
                                W_try[rr, c] = np.random.choice([-1, 1])
                    if verify(U_try, V_try, W_try) < 1e-9:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = target_rank
                        found = True
                        break
                if found:
                    break
            if found:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
        except Exception:
            pass

    # ---- Fallback: return the rank-50 decomposition ----
    return {"U": U, "V": V, "W": W}
