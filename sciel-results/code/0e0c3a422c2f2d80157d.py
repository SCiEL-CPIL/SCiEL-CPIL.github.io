def construct():
    import time
    start = time.time()
    deadline = start + 115.0

    n, m, p = 2, 2, 8

    # Build the rank-28 Strassen solution
    def _strassen_28():
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1

            # Term 1: m1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
            U[0][col] = 1.0
            U[3][col] = 1.0
            V[0*p + k0][col] = 1.0
            V[1*p + k1][col] = 1.0
            W[k0*n + 0][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1

            # Term 2: m2 = (A[1,0]+A[1,1]) * B[0,k0]
            U[2][col] = 1.0
            U[3][col] = 1.0
            V[0*p + k0][col] = 1.0
            W[k0*n + 1][col] = 1.0
            W[k1*n + 1][col] = -1.0
            col += 1

            # Term 3: m3 = A[0,0] * (B[0,k1]-B[1,k1])
            U[0][col] = 1.0
            V[0*p + k1][col] = 1.0
            V[1*p + k1][col] = -1.0
            W[k1*n + 0][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1

            # Term 4: m4 = A[1,1] * (B[1,k0]-B[0,k0])
            U[3][col] = 1.0
            V[1*p + k0][col] = 1.0
            V[0*p + k0][col] = -1.0
            W[k0*n + 0][col] = 1.0
            W[k0*n + 1][col] = 1.0
            col += 1

            # Term 5: m5 = (A[0,0]+A[0,1]) * B[1,k1]
            U[0][col] = 1.0
            U[1][col] = 1.0
            V[1*p + k1][col] = 1.0
            W[k0*n + 0][col] = -1.0
            W[k1*n + 0][col] = 1.0
            col += 1

            # Term 6: m6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
            U[2][col] = 1.0
            U[0][col] = -1.0
            V[0*p + k0][col] = 1.0
            V[0*p + k1][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1

            # Term 7: m7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
            U[1][col] = 1.0
            U[3][col] = -1.0
            V[1*p + k0][col] = 1.0
            V[1*p + k1][col] = 1.0
            W[k0*n + 0][col] = 1.0
            col += 1

        return U, V, W, rank

    U28, V28, W28, rank28 = _strassen_28()

    # Verify the rank-28 solution
    def _verify(U, V, W, rank):
        import numpy as np
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(actual - 1.0) > 1e-8:
                        return False
        return True

    if not _verify(U28, V28, W28, rank28):
        return {"U": U28, "V": V28, "W": W28}

    # Try to reduce rank by finding a rank-(r-1) decomposition
    import numpy as np

    def _try_reduce_rank(U, V, W, rank, max_tries=50):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        for remove_idx in range(rank):
            R = np.zeros((n, m, p), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i, j, k] = 1.0
                        for c in range(rank):
                            if c != remove_idx:
                                R[i, j, k] -= U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]

            nonzero = np.nonzero(np.abs(R) > 1e-10)
            if len(nonzero[0]) == 0:
                new_rank = rank - 1
                U_new = [U[r][:remove_idx] + U[r][remove_idx+1:] for r in range(n*m)]
                V_new = [V[r][:remove_idx] + V[r][remove_idx+1:] for r in range(m*p)]
                W_new = [W[r][:remove_idx] + W[r][remove_idx+1:] for r in range(p*n)]
                return U_new, V_new, W_new, new_rank

            idxs = nonzero
            num_eq = len(idxs[0])

            if num_eq < 1:
                continue

            num_vars = n*m + m*p + p*n
            A_mat = np.zeros((num_eq, num_vars), dtype=np.float64)
            b_vec = np.zeros(num_eq, dtype=np.float64)

            valid = True
            for eq_idx in range(num_eq):
                i_val = idxs[0][eq_idx]
                j_val = idxs[1][eq_idx]
                k_val = idxs[2][eq_idx]
                val = R[i_val, j_val, k_val]
                if abs(val) < 1e-15:
                    valid = False
                    break
                A_mat[eq_idx, i_val*m + j_val] = 1.0
                A_mat[eq_idx, n*m + j_val*p + k_val] = 1.0
                A_mat[eq_idx, n*m + m*p + k_val*n + i_val] = 1.0
                b_vec[eq_idx] = np.log(abs(val))

            if not valid:
                continue

            try:
                sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                continue

            if residuals.size > 0 and residuals[0] > 1e-6:
                continue
            pred = A_mat @ sol
            if np.max(np.abs(pred - b_vec)) > 1e-6:
                continue

            a_mag = np.exp(sol[:n*m]).reshape(n, m)
            b_mag = np.exp(sol[n*m:n*m+m*p]).reshape(m, p)
            c_mag = np.exp(sol[n*m+m*p:]).reshape(p, n)

            mag_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(R[i,j,k]) > 1e-10:
                            if abs(a_mag[i,j]*b_mag[j,k]*c_mag[k,i] - abs(R[i,j,k])) > 1e-6:
                                mag_ok = False
                                break
                    if not mag_ok:
                        break
                if not mag_ok:
                    break

            if not mag_ok:
                continue

            xor_A = np.zeros((num_eq, num_vars), dtype=np.int8)
            xor_b = np.zeros(num_eq, dtype=np.int8)

            for eq_idx in range(num_eq):
                i_val = idxs[0][eq_idx]
                j_val = idxs[1][eq_idx]
                k_val = idxs[2][eq_idx]
                xor_A[eq_idx, i_val*m + j_val] = 1
                xor_A[eq_idx, n*m + j_val*p + k_val] = 1
                xor_A[eq_idx, n*m + m*p + k_val*n + i_val] = 1
                xor_b[eq_idx] = 1 if R[i_val, j_val, k_val] < 0 else 0

            aug = np.column_stack([xor_A, xor_b]).copy()
            nrows, ncols = aug.shape
            ncols = num_vars + 1

            rank_gf2 = 0
            for col in range(num_vars):
                pivot = -1
                for row in range(rank_gf2, nrows):
                    if aug[row, col] == 1:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                aug[[rank_gf2, pivot]] = aug[[pivot, rank_gf2]]
                for row in range(nrows):
                    if row != rank_gf2 and aug[row, col] == 1:
                        aug[row] = (aug[row] + aug[rank_gf2]) % 2
                rank_gf2 += 1

            consistent = True
            for row in range(rank_gf2, nrows):
                if aug[row, num_vars] == 1:
                    consistent = False
                    break

            if not consistent:
                continue

            sol_gf2 = np.zeros(num_vars, dtype=np.int8)
            for col in range(num_vars):
                pivot_row = -1
                for row in range(rank_gf2):
                    if aug[row, col] == 1:
                        is_leading = True
                        for prev_col in range(col):
                            if aug[row, prev_col] == 1:
                                is_leading = False
                                break
                        if is_leading:
                            pivot_row = row
                            break
                if pivot_row >= 0:
                    sol_gf2[col] = aug[pivot_row, num_vars]

            s_a = np.where(sol_gf2[:n*m] == 1, -1, 1).reshape(n, m)
            s_b = np.where(sol_gf2[n*m:n*m+m*p] == 1, -1, 1).reshape(m, p)
            s_c = np.where(sol_gf2[n*m+m*p:] == 1, -1, 1).reshape(p, n)

            a_full = a_mag * s_a
            b_full = b_mag * s_b
            c_full = c_mag * s_c

            sign_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = R[i, j, k]
                        actual = a_full[i,j] * b_full[j,k] * c_full[k,i]
                        if abs(actual - expected) > 1e-6:
                            sign_ok = False
                            break
                    if not sign_ok:
                        break
                if not sign_ok:
                    break

            if not sign_ok:
                continue

            new_rank = rank - 1
            U_new = [[0.0] * new_rank for _ in range(n * m)]
            V_new = [[0.0] * new_rank for _ in range(m * p)]
            W_new = [[0.0] * new_rank for _ in range(p * n)]

            for c in range(rank):
                if c == remove_idx:
                    continue
                new_c = c if c < remove_idx else c - 1
                for i in range(n):
                    for j in range(m):
                        U_new[i*m+j][new_c] = U_np[i*m+j, c]
                for j in range(m):
                    for k in range(p):
                        V_new[j*p+k][new_c] = V_np[j*p+k, c]
                for k in range(p):
                    for i in range(n):
                        W_new[k*n+i][new_c] = W_np[k*n+i, c]

            for i in range(n):
                for j in range(m):
                    U_new[i*m+j][new_rank-1] = a_full[i,j]
            for j in range(m):
                for k in range(p):
                    V_new[j*p+k][new_rank-1] = b_full[j,k]
            for k in range(p):
                for i in range(n):
                    W_new[k*n+i][new_rank-1] = c_full[k,i]

            return U_new, V_new, W_new, new_rank

        return None

    best_U, best_V, best_W, best_rank = U28, V28, W28, rank28

    U_curr, V_curr, W_curr = U28, V28, W28
    rank_curr = rank28

    while rank_curr > 16:
        if time.time() > deadline - 10.0:
            break

        try:
            result = _try_reduce_rank(U_curr, V_curr, W_curr, rank_curr)
        except Exception:
            result = None

        if result is None:
            break
        U_curr, V_curr, W_curr, rank_curr = result
        best_U, best_V, best_W, best_rank = U_curr, V_curr, W_curr, rank_curr

        if not _verify(U_curr, V_curr, W_curr, rank_curr):
            break

    if best_rank < rank28:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": U28, "V": V28, "W": W28}
