import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110

    # Build target tensor T with indices (i*m+j, j*p+k, k*n+i)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, tol=1e-9):
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def check_half_int(M, tol=1e-6):
        for val in M.flatten():
            if abs(2*val - round(2*val)) > tol:
                return False
        return True

    def round_half_int(M):
        return np.round(2*M) / 2.0

    def get_schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_s, V_s, W_s = get_schoolbook()
    best_rank = n * m * p
    best_result = (U_s, V_s, W_s)

    def try_rank(r, max_iter=80, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        norm = np.sqrt(r)
        U /= norm
        V /= norm
        W /= norm

        best_err = float('inf')

        for it in range(max_iter):
            if time.time() > deadline:
                break

            # Update U
            M_u = np.einsum('jc,kc->jkc', V, W).reshape(r, -1)
            T_flat = T.reshape(n*m, -1)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_u.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass

            # Update V
            M_v = np.einsum('ic,kc->ick', U, W).reshape(r, -1)
            T_flat_v = T.transpose(1, 0, 2).reshape(m*p, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass

            # Update W
            M_w = np.einsum('ic,jc->ijc', U, V).reshape(r, -1)
            T_flat_w = T.transpose(2, 0, 1).reshape(p*n, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass

            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                if err < 1e-6:
                    U_r = round_half_int(U)
                    V_r = round_half_int(V)
                    W_r = round_half_int(W)
                    if verify(U_r, V_r, W_r, tol=1e-5):
                        return U_r, V_r, W_r

        return None

    # Try ranks from 39 down to 30
    ranks_to_try = list(range(39, 29, -1))

    for r in ranks_to_try:
        if time.time() > deadline:
            break

        found = False
        for seed in range(15):
            if time.time() > deadline:
                break
            res = try_rank(r, max_iter=60, seed=seed)
            if res is not None:
                U, V, W = res
                if verify(U, V, W, tol=1e-6) and check_half_int(U) and check_half_int(V) and check_half_int(W):
                    best_rank = r
                    best_result = (U, V, W)
                    found = True
                    break

        if found:
            continue

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
