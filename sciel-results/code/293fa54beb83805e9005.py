"""Strassen-based rank-55 decomposition with structured rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[i,j,k] = 1 at (i*m+j, j*p+k, k*n+i)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_flat_nm = T.reshape(nm, mp * pn)
    T_flat_mp = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_flat_pn = T.transpose(2, 0, 1).reshape(pn, nm * mp)

    def verify(U, V, W, tol=1e-10):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T_recon - T))

    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        # Part A: j=0 (20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of 2x2 (35 terms)
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1

            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1

            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1

            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1

            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1

            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1

            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        return U, V, W, rank

    U_best, V_best, W_best, rank_best = rank55_decomposition()

    def try_rank_als(r, U_init, V_init, W_init, n_iters=50, tol=1e-12):
        """ALS with high-precision polish pass."""
        U_c = U_init.copy()
        V_c = V_init.copy()
        W_c = W_init.copy()

        for it in range(n_iters):
            if time.time() > deadline - 5:
                return None

            # Solve for U
            M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
            try:
                U_new = T_flat_nm @ np.linalg.pinv(M.T, rcond=1e-15)
                if np.linalg.norm(U_new - U_c) < tol:
                    break
                U_c = U_new
            except:
                return None

            # Solve for V
            N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
            try:
                V_new = T_flat_mp @ np.linalg.pinv(N.T, rcond=1e-15)
                if np.linalg.norm(V_new - V_c) < tol:
                    break
                V_c = V_new
            except:
                return None

            # Solve for W
            P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
            try:
                W_new = T_flat_pn @ np.linalg.pinv(P.T, rcond=1e-15)
                if np.linalg.norm(W_new - W_c) < tol:
                    break
                W_c = W_new
            except:
                return None

        # Check convergence
        T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
        error = np.max(np.abs(T_check - T))

        if error < 1e-8:
            # Try rounding to half-integers
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2

            T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.max(np.abs(T_check2 - T)) < 1e-10:
                return U_r, V_r, W_r, r
            elif error < 1e-10:
                # Check if entries are close to half-integers
                U_close = np.allclose(U_c, np.round(U_c * 2) / 2, atol=1e-6)
                V_close = np.allclose(V_c, np.round(V_c * 2) / 2, atol=1e-6)
                W_close = np.allclose(W_c, np.round(W_c * 2) / 2, atol=1e-6)
                if U_close and V_close and W_close:
                    return U_r, V_r, W_r, r
                else:
                    return U_c, V_c, W_c, r

        return None

    def local_search(U, V, W, r, n_steps=100):
        """Local search to find exact half-integer solution."""
        U_c = U.copy()
        V_c = V.copy()
        W_c = W.copy()

        for step in range(n_steps):
            if time.time() > deadline - 5:
                return None

            # Try to zero out small entries
            for mat_name, mat in [('U', U_c), ('V', V_c), ('W', W_c)]:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if abs(mat[i, j]) < 0.01 and abs(mat[i, j]) > 1e-10:
                            # Try setting to 0
                            old_val = mat[i, j]
                            mat[i, j] = 0

                            T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
                            error = np.max(np.abs(T_check - T))

                            if error > 1e-8:
                                mat[i, j] = old_val
                            else:
                                # Keep zeroed
                                pass

            # Try small perturbations to find better solution
            if step % 10 == 0:
                # Round to half-integers
                U_r = np.round(U_c * 2) / 2
                V_r = np.round(V_c * 2) / 2
                W_r = np.round(W_c * 2) / 2

                T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                error = np.max(np.abs(T_check - T))

                if error < 1e-10:
                    return U_r, V_r, W_r, r

        return None

    # Structured rank-reduction search
    if time.time() < deadline - 15:
        for r in range(54, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 10:
                break

            found = False
            n_trials = 30 if r >= 50 else 15

            for trial in range(n_trials):
                if time.time() > deadline - 10:
                    break

                np.random.seed(trial * 613 + r * 29)

                if trial < 6:
                    # Subsample from rank-55 with small noise
                    idx = np.random.choice(55, r, replace=False)
                    U_c = U_best[:, idx].copy() + 0.01 * np.random.randn(nm, r)
                    V_c = V_best[:, idx].copy() + 0.01 * np.random.randn(mp, r)
                    W_c = W_best[:, idx].copy() + 0.01 * np.random.randn(pn, r)
                elif trial < 12:
                    # Full rank-55 truncated with larger noise
                    U_c = U_best[:, :r].copy() + 0.05 * np.random.randn(nm, r)
                    V_c = V_best[:, :r].copy() + 0.05 * np.random.randn(mp, r)
                    W_c = W_best[:, :r].copy() + 0.05 * np.random.randn(pn, r)
                elif trial < 18:
                    # Mixed: 70% structured + 30% random
                    U_c = 0.7 * U_best[:, :r] + 0.3 * np.random.randn(nm, r)
                    V_c = 0.7 * V_best[:, :r] + 0.3 * np.random.randn(mp, r)
                    W_c = 0.7 * W_best[:, :r] + 0.3 * np.random.randn(pn, r)
                elif trial < 24:
                    # Pure random
                    U_c = np.random.randn(nm, r)
                    V_c = np.random.randn(mp, r)
                    W_c = np.random.randn(pn, r)
                else:
                    # Scaled random
                    U_c = 0.3 * np.random.randn(nm, r)
                    V_c = 0.3 * np.random.randn(mp, r)
                    W_c = 0.3 * np.random.randn(pn, r)

                # ALS iterations
                result = try_rank_als(r, U_c, V_c, W_c, n_iters=40)
                if result is not None:
                    U_r, V_r, W_r, rank_r = result

                    # Local search to polish
                    result2 = local_search(U_r, V_r, W_r, rank_r, n_steps=50)
                    if result2 is not None:
                        U_best, V_best, W_best, rank_best = result2
                        found = True
                        break
                    else:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, rank_r
                        found = True
                        break

            if found:
                break

    # Final verification
    final_err = verify(U_best, V_best, W_best)
    if final_err > 1e-8:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
