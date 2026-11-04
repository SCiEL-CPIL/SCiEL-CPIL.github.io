"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _t0 = time.time()
    _deadline = _t0 + 145
    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_M(U_mat, V_mat, W_mat, r):
        # M has shape (n*m*p, r); M[ijk, c] = U[i,j,c]*V[j,k,c]*W[k,i,c]
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        M = np.einsum('ijc,jkc,kic->ijkc', u, v, w)
        return M.reshape(n * m * p, r)

    # ---- Build the rank-66 block-structured decomposition ----
    U66 = np.zeros((n * m, 66))
    V66 = np.zeros((m * p, 66))
    W66 = np.zeros((p * n, 66))
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        U66[A00, col] += 1; U66[A11, col] += 1
        V66[B00, col] += 1; V66[B11, col] += 1
        W66[C00, col] += 1; W66[C11, col] += 1
        col += 1

        U66[A10, col] += 1; U66[A11, col] += 1
        V66[B00, col] += 1
        W66[C10, col] += 1; W66[C11, col] -= 1
        col += 1

        U66[A00, col] += 1
        V66[B01, col] += 1; V66[B11, col] -= 1
        W66[C01, col] += 1; W66[C11, col] += 1
        col += 1

        U66[A11, col] += 1
        V66[B10, col] += 1; V66[B00, col] -= 1
        W66[C00, col] += 1; W66[C10, col] += 1
        col += 1

        U66[A00, col] += 1; U66[A01, col] += 1
        V66[B11, col] += 1
        W66[C00, col] -= 1; W66[C01, col] += 1
        col += 1

        U66[A10, col] += 1; U66[A00, col] -= 1
        V66[B00, col] += 1; V66[B01, col] += 1
        W66[C11, col] += 1
        col += 1

        U66[A01, col] += 1; U66[A11, col] -= 1
        V66[B10, col] += 1; V66[B11, col] += 1
        W66[C00, col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U66[A0, col] += 1; V66[B00, col] += 1; W66[C0, col] += 1
        col += 1
        U66[A1, col] += 1; V66[B10, col] += 1; W66[C0, col] += 1
        col += 1
        U66[A0, col] += 1; V66[B01, col] += 1; W66[C1, col] += 1
        col += 1
        U66[A1, col] += 1; V66[B11, col] += 1; W66[C1, col] += 1
        col += 1

    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    # Fallback to schoolbook if the block structure is wrong
    if not verify(U66, V66, W66, 66):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, cc] = 1
                    V[j * p + k, cc] = 1
                    W[k * n + i, cc] = 1
                    cc += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Rank reduction via nullspace of the flattening matrix ----
    def try_reduce(U_cur, V_cur, W_cur, r):
        U_next, V_next, W_next, r_next = U_cur, V_cur, W_cur, r
        for iteration in range(30):
            if time.time() > _deadline - 12:
                break
            M = build_M(U_next, V_next, W_next, r_next)
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            except Exception:
                break
            tol = 1e-9
            null_dim = int(np.sum(s < tol))
            if null_dim == 0:
                break
            nullspace = Vt[r_next - null_dim:, :]

            best = None
            best_rank = r_next
            for nv_idx in range(null_dim):
                if time.time() > _deadline - 12:
                    break
                null_vec = nullspace[nv_idx, :]
                abs_nv = np.abs(null_vec)
                order = np.argsort(-abs_nv)
                for elim_idx in order[:min(8, r_next)]:
                    if abs(null_vec[elim_idx]) < 1e-9:
                        continue
                    new_keep = [c for c in range(r_next) if c != elim_idx]
                    # For each of the three matrices as the scaling target,
                    # scale the surviving columns by (1 - null_vec[c]/null_vec[elim]).
                    for which in range(3):
                        U_test = U_next[:, new_keep].copy()
                        V_test = V_next[:, new_keep].copy()
                        W_test = W_next[:, new_keep].copy()
                        target = [U_test, V_test, W_test][which]
                        for ni, oi in enumerate(new_keep):
                            gamma = 1 - null_vec[oi] / null_vec[elim_idx]
                            target[:, ni] = target[:, ni] * gamma
                        if verify(U_test, V_test, W_test, len(new_keep)):
                            if len(new_keep) < best_rank:
                                best_rank = len(new_keep)
                                best = (U_test, V_test, W_test)
            if best is None:
                break
            U_next, V_next, W_next = best
            r_next = best_rank
        return U_next, V_next, W_next, r_next

    U_final, V_final, W_final, r_final = try_reduce(U66, V66, W66, 66)

    # ---- Final high-precision polish: tighten entries to exact halves ----
    def polish(U_mat, V_mat, W_mat):
        U_mat = np.round(U_mat * 2) / 2
        V_mat = np.round(V_mat * 2) / 2
        W_mat = np.round(W_mat * 2) / 2
        return U_mat, V_mat, W_mat

    U_final, V_final, W_final = polish(U_final, V_final, W_final)
    if not verify(U_final, V_final, W_final, U_final.shape[1]):
        # revert polish if it broke exactness
        U_final, V_final, W_final, r_final = try_reduce(U66, V66, W66, 66)

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
