"""Strassen-based rank-55 decomposition with structured rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[i,j,k] = 1 at (i*m+j, j*p+k, k*n+i)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_flat_nm = T.reshape(nm, mp * pn)
    T_flat_mp = T.reshape(mp, nm * pn)
    T_flat_pn = T.reshape(pn, nm * mp)

    def verify(U, V, W, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

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

    # Structured rank-reduction search
    if time.time() < deadline - 10:
        for r in range(54, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 5:
                break

            found = False
            # More diverse initialization strategies
            n_trials = 40 if r >= 50 else 20

            for trial in range(n_trials):
                if time.time() > deadline - 5:
                    break

                np.random.seed(trial * 613 + r * 29)

                if trial < 8:
                    # Subsample from rank-55 with small noise
                    idx = np.random.choice(55, r, replace=False)
                    U_c = U_best[:, idx].copy() + 0.02 * np.random.randn(nm, r)
                    V_c = V_best[:, idx].copy() + 0.02 * np.random.randn(mp, r)
                    W_c = W_best[:, idx].copy() + 0.02 * np.random.randn(pn, r)
                elif trial < 16:
                    # Full rank-55 truncated with larger noise
                    U_c = U_best[:, :r].copy() + 0.1 * np.random.randn(nm, r)
                    V_c = V_best[:, :r].copy() + 0.1 * np.random.randn(mp, r)
                    W_c = W_best[:, :r].copy() + 0.1 * np.random.randn(pn, r)
                elif trial < 24:
                    # Mixed: 60% structured + 40% random
                    U_c = 0.6 * U_best[:, :r] + 0.4 * np.random.randn(nm, r)
                    V_c = 0.6 * V_best[:, :r] + 0.4 * np.random.randn(mp, r)
                    W_c = 0.6 * W_best[:, :r] + 0.4 * np.random.randn(pn, r)
                elif trial < 32:
                    # Pure random
                    U_c = np.random.randn(nm, r)
                    V_c = np.random.randn(mp, r)
                    W_c = np.random.randn(pn, r)
                else:
                    # Scaled random
                    U_c = 0.5 * np.random.randn(nm, r)
                    V_c = 0.5 * np.random.randn(mp, r)
                    W_c = 0.5 * np.random.randn(pn, r)

                # ALS iterations with adaptive refinement
                for it in range(30):
                    if time.time() > deadline - 5:
                        break

                    # Solve for U
                    M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                    try:
                        U_new = T_flat_nm @ np.linalg.pinv(M.T)
                        if np.linalg.norm(U_new - U_c) < 1e-13:
                            break
                        U_c = U_new
                    except:
                        break

                    # Solve for V
                    N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                    try:
                        V_new = T_flat_mp @ np.linalg.pinv(N.T)
                        if np.linalg.norm(V_new - V_c) < 1e-13:
                            break
                        V_c = V_new
                    except:
                        break

                    # Solve for W
                    P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                    try:
                        W_new = T_flat_pn @ np.linalg.pinv(P.T)
                        if np.linalg.norm(W_new - W_c) < 1e-13:
                            break
                        W_c = W_new
                    except:
                        break

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
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        found = True
                        break
                    elif error < 1e-10:
                        U_best, V_best, W_best, rank_best = U_c, V_c, W_c, r
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
