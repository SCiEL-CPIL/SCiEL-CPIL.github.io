import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    n, m, p = 2, 4, 7

    # Build the Strassen block decomposition (rank 50)
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    def get_strassen_222():
        r = 7
        U_s = np.zeros((4, r))
        V_s = np.zeros((4, r))
        W_s = np.zeros((4, r))
        # M1 = (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        # M2 = (A10+A11)B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = -1
        # M3 = A00(B01-B11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[2, 2] = 1; W_s[3, 2] = 1
        # M4 = A11(B10-B00)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1
        # M5 = (A00+A01)B11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[2, 4] = 1
        # M6 = (A10-A00)(B00+B01)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        # M7 = (A01-A11)(B10+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        return U_s, V_s, W_s

    def get_school_221():
        r = 4
        U_s = np.zeros((4, r))
        V_s = np.zeros((2, r))
        W_s = np.zeros((2, r))
        for c in range(r):
            i = c // 2
            j = c % 2
            U_s[i * 2 + j, c] = 1.0
            V_s[j, c] = 1.0
            W_s[i, c] = 1.0
        return U_s, V_s, W_s

    col_offset = 0
    for j_off in [0, 2]:
        # p blocks: [0,2), [2,4), [4,6) use Strassen; [6,7) uses schoolbook
        for k_off in [0, 2, 4]:
            U_s, V_s, W_s = get_strassen_222()
            r_sub = U_s.shape[1]
            for i in range(2):
                for j in range(2):
                    U[i * m + (j_off + j), col_offset:col_offset + r_sub] = U_s[i * 2 + j, :]
            for j in range(2):
                for k in range(2):
                    V[(j_off + j) * p + (k_off + k), col_offset:col_offset + r_sub] = V_s[j * 2 + k, :]
            for k in range(2):
                for i in range(2):
                    W[(k_off + k) * n + i, col_offset:col_offset + r_sub] = W_s[k * 2 + i, :]
            col_offset += r_sub

        # p in [6, 7): Schoolbook 2x2x1
        U_s, V_s, W_s = get_school_221()
        r_sub = U_s.shape[1]
        for i in range(2):
            for j in range(2):
                U[i * m + (j_off + j), col_offset:col_offset + r_sub] = U_s[i * 2 + j, :]
        for j in range(2):
            V[(j_off + j) * p + 6, col_offset:col_offset + r_sub] = V_s[j, :]
        for i in range(2):
            W[6 * n + i, col_offset:col_offset + r_sub] = W_s[i, :]
        col_offset += r_sub

    # Build target tensor for verification
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Verify base solution
    def tensor_from(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

    def check(U, V, W, T, tol=1e-6):
        R = tensor_from(U, V, W)
        return np.allclose(R, T, atol=tol, rtol=tol)

    assert check(U, V, W, T), "Base Strassen solution failed verification"

    # Alternating minimization to try to find lower-rank representation
    # Strategy: try ranks from 49 down to some minimum, using the rank-50 solution as init
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = total_rank

    a, b, c = n * m, m * p, p * n  # 8, 28, 14

    def alt_minimize(U_init, V_init, W_init, T, r, max_iter=200, tol=1e-8):
        """Try to find rank-r decomposition starting from given init."""
        if r >= U_init.shape[1]:
            # Use first r columns
            U_curr = U_init[:, :r].copy()
            V_curr = V_init[:, :r].copy()
            W_curr = W_init[:, :r].copy()
        else:
            # Perturb and truncate
            rng = np.random.RandomState(42)
            U_curr = U_init[:, :r].copy() + rng.randn(a, r) * 0.05
            V_curr = V_init[:, :r].copy() + rng.randn(b, r) * 0.05
            W_curr = W_init[:, :r].copy() + rng.randn(c, r) * 0.05

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None

            # Optimize W: T[i,j,k] = sum_d U[i,d]*V[j,d]*W[k,d]
            # For each k: T[:,:,k] = sum_d W[k,d] * outer(U[:,d], V[:,d])
            # Linear system: for each k, (a*b) equations, r unknowns
            D_w = np.zeros((a * b * c, r))
            for d in range(r):
                outer_d = np.outer(U_curr[:, d], V_curr[:, d]).reshape(-1)
                for k in range(c):
                    D_w[k * a * b:(k+1) * a * b, d] = outer_d
            T_flat = T.transpose(2, 0, 1).reshape(c, a * b).T.reshape(-1)
            W_new, _, _, _ = np.linalg.lstsq(D_w, T_flat, rcond=None)
            W_curr = W_new.reshape(c, r)

            # Optimize U: T[i,j,k] = sum_d U[i,d]*V[j,d]*W[k,d]
            D_u = np.zeros((a * b * c, r))
            for d in range(r):
                outer_d = np.outer(V_curr[:, d], W_curr[:, d]).reshape(-1)
                for i in range(a):
                    D_u[i * b * c:(i+1) * b * c, d] = outer_d
            T_flat_u = T.transpose(0, 1, 2).reshape(a, b * c).T.reshape(-1)
            U_new, _, _, _ = np.linalg.lstsq(D_u, T_flat_u, rcond=None)
            U_curr = U_new.reshape(a, r)

            # Optimize V: T[i,j,k] = sum_d U[i,d]*V[j,d]*W[k,d]
            D_v = np.zeros((a * b * c, r))
            for d in range(r):
                outer_d = np.outer(U_curr[:, d], W_curr[:, d]).reshape(-1)
                for j in range(b):
                    D_v[j * a * c:(j+1) * a * c, d] = outer_d
            T_flat_v = T.transpose(1, 0, 2).reshape(b, a * c).T.reshape(-1)
            V_new, _, _, _ = np.linalg.lstsq(D_v, T_flat_v, rcond=None)
            V_curr = V_new.reshape(b, r)

            # Check convergence
            R = tensor_from(U_curr, V_curr, W_curr)
            err = np.linalg.norm(R - T) / np.linalg.norm(T)
            if err < tol:
                break

        # Final check
        R = tensor_from(U_curr, V_curr, W_curr)
        if np.allclose(R, T, atol=1e-5, rtol=1e-5):
            return U_curr, V_curr, W_curr
        return None

    # Try to reduce rank
    for r_try in range(total_rank - 1, 30, -1):
        if time.time() > deadline - 10:
            break

        result = alt_minimize(U, V, W, T, r_try, max_iter=100)
        if result is not None:
            U_curr, V_curr, W_curr = result
            # Round to half-integers if possible
            U_rounded = np.round(U_curr * 2) / 2
            V_rounded = np.round(V_curr * 2) / 2
            W_rounded = np.round(W_curr * 2) / 2
            if check(U_rounded, V_rounded, W_rounded, T, tol=1e-4):
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = r_try
            else:
                best_U, best_V, best_W = U_curr, V_curr, W_curr
                best_rank = r_try
            break  # Found a valid lower rank, stop

    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U[:, :best_rank]]
    V_list = [[float(x) for x in row] for row in best_V[:, :best_rank]]
    W_list = [[float(x) for x in row] for row in best_W[:, :best_rank]]

    return {"U": U_list, "V": V_list, "W": W_list}
