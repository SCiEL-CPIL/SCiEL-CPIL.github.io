"""Matrix multiplication 4,4,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    time_limit = 145  # Leave margin before 150s kill

    # ------------------------------------------------------------------
    # Helper: verification via einsum
    # ------------------------------------------------------------------
    def verify_numpy(U, V, W, r):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.max(np.abs(T - 1.0))

    # ------------------------------------------------------------------
    # Build Strassen 4x4x4 decomposition (rank 49)
    # ------------------------------------------------------------------
    def build_strassen_4x4x4():
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))

        # Level-1 Strassen for 4x4x4 (7 products of 2x2x2)
        l1 = [
            [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)],
        ]
        # Full Strassen 2x2 patterns
        strassen2 = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ]

        # Level-1: 7 block-level Strassen products
        l1_blocks = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ]

        prod_idx = 0
        for a_blks, b_blks, c_blks in l1_blocks:
            for a_coeff, b_coeff, c_coeff in strassen2:
                for (bi_a, bj_a, ca_blk) in a_blks:
                    for (bj_b, bk_b, cb_blk) in b_blks:
                        for (bi_c, bk_c, cc_blk) in c_blks:
                            if bi_a != bi_c or bj_a != bj_b or bk_b != bk_c:
                                continue
                            bi, bj, bk = bi_a, bj_a, bk_b
                            for (di, dj, ca) in a_coeff:
                                for (dj2, dk, cb) in b_coeff:
                                    if dj != dj2:
                                        continue
                                    for (di2, dk2, cc) in c_coeff:
                                        if di != di2 or dk != dk2:
                                            continue
                                        i = 2*bi + di
                                        j = 2*bj + dj
                                        k = 2*bk + dk
                                        coeff = ca * cb * cc * ca_blk * cb_blk * cc_blk
                                        U[i*4+j, prod_idx] += coeff
                                        V[j*4+k, prod_idx] += coeff
                                        W[k*4+i, prod_idx] += coeff
                prod_idx += 1

        return U, V, W

    # Build Strassen 4x4x4
    U_str, V_str, W_str = build_strassen_4x4x4()

    # ------------------------------------------------------------------
    # Build the full 4x4x5 decomposition:
    # Part 1: 4x4x4 (Strassen, rank 49) - uses first 4 columns of B
    # Part 2: 4x4x1 (schoolbook, rank 16) - uses last column of B
    # ------------------------------------------------------------------
    rank_total = 49 + 16
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))  # m*p = 4*5 = 20
    W = np.zeros((20, rank_total))  # p*n = 5*4 = 20

    U[:, :49] = U_str
    V[:16, :49] = V_str
    W[:16, :49] = W_str

    # Part 2: Schoolbook 4x4x1 for k=4
    t = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j, 49+t] = 1
            V[j*4+4, 49+t] = 1
            W[4*4+i, 49+t] = 1
            t += 1

    # Verify
    err = verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank_total)
    if err > 1e-6:
        # Fallback to schoolbook
        U = np.zeros((16, 80))
        V = np.zeros((20, 80))
        W = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        rank_total = 80
        err = verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank_total)

    # ------------------------------------------------------------------
    # Try to reduce rank by optimizing the combined decomposition
    # ------------------------------------------------------------------
    def try_reduce_rank(r, max_iters=3000, time_budget=8):
        """Try to find a rank-r decomposition by optimizing from the current one."""
        if r >= rank_total:
            return None
        if time.time() - start_time > time_limit - 10:
            return None

        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        total_size = size_u + size_v + size_w

        # Initialize from the current solution, truncated/padded
        U_init = np.zeros((n*m, r))
        V_init = np.zeros((m*p, r))
        W_init = np.zeros((p*n, r))
        min_r = min(r, rank_total)
        U_init[:, :min_r] = U[:, :min_r]
        V_init[:, :min_r] = V[:, :min_r]
        W_init[:, :min_r] = W[:, :min_r]

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

        def objective(x):
            if time.time() - start_time > time_limit:
                return 1e10
            u_flat = x[:size_u]
            v_flat = x[size_u:size_u + size_v]
            w_flat = x[size_u + size_v:]

            U_mat = u_flat.reshape(n * m, r)
            V_mat = v_flat.reshape(m * p, r)
            W_mat = w_flat.reshape(p * n, r)

            U3 = U_mat.reshape(n, m, r)
            V3 = V_mat.reshape(m, p, r)
            W3 = W_mat.reshape(p, n, r)

            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            error = np.sum((T - 1.0) ** 2)
            return error

        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': max_iters, 'maxfun': max_iters * 5})
            if res.fun < 1e-6:
                x_opt = res.x
                U_mat = x_opt[:size_u].reshape(n * m, r)
                V_mat = x_opt[size_u:size_u + size_v].reshape(m * p, r)
                W_mat = x_opt[size_u + size_v:].reshape(p * n, r)

                # Round to nearest 0.5
                U_list = [[round(U_mat[i][c] * 2) / 2.0 for c in range(r)] for i in range(n*m)]
                V_list = [[round(V_mat[i][c] * 2) / 2.0 for c in range(r)] for i in range(m*p)]
                W_list = [[round(W_mat[i][c] * 2) / 2.0 for c in range(r)] for i in range(p*n)]

                if verify_numpy(U_list, V_list, W_list, r) < 1e-6:
                    return U_list, V_list, W_list
        except Exception:
            pass
        return None

    # Try to reduce rank from 65 down to 60
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    best_rank = rank_total

    for r in range(64, 59, -1):
        if time.time() - start_time > time_limit - 15:
            break
        result = try_reduce_rank(r)
        if result:
            best_U, best_V, best_W = result
            best_rank = r
            break

    # Final verification
    err = verify_numpy(best_U, best_V, best_W, best_rank)
    if err > 1e-6:
        # Fall back to the verified 65-rank solution
        best_U = U.tolist()
        best_V = V.tolist()
        best_W = W.tolist()
        best_rank = rank_total
        err = verify_numpy(best_U, best_V, best_W, best_rank)
        if err > 1e-6:
            # Final fallback: schoolbook
            best_U = [[0.0] * 80 for _ in range(16)]
            best_V = [[0.0] * 80 for _ in range(20)]
            best_W = [[0.0] * 80 for _ in range(20)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(5):
                        best_U[i*4+j][col] = 1.0
                        best_V[j*5+k][col] = 1.0
                        best_W[k*4+i][col] = 1.0
                        col += 1
            best_rank = 80

    return {"U": best_U, "V": best_V, "W": best_W}
