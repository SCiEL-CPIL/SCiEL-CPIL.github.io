"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s kill

    # ---------- Build schoolbook rank-48 decomposition ----------
    rank0 = n * m * p  # 48
    U = np.zeros((n * m, rank0))
    V = np.zeros((m * p, rank0))
    W = np.zeros((p * n, rank0))

    r = 0
    for i_A in range(n):
        for j in range(m):
            for k_B in range(p):
                i = i_A * m + j
                j_idx = j * p + k_B
                k = k_B * n + i_A
                U[i, r] = 1.0
                V[j_idx, r] = 1.0
                W[k, r] = 1.0
                r += 1

    # ---------- Helper functions ----------
    def tensor_of_column(U_col, V_col, W_col):
        """Outer product U_col * V_col * W_col as flat vector."""
        return np.outer(np.outer(U_col, V_col), W_col).flatten()

    def full_tensor(U_mat, V_mat, W_mat):
        return np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)

    def target_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    i = i_A * m + j
                    j_idx = j * p + k_B
                    k = k_B * n + i_A
                    T[i, j_idx, k] = 1.0
        return T

    T_target = target_tensor()

    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        T_comp = full_tensor(U_mat, V_mat, W_mat)
        return np.max(np.abs(T_comp - T_target)) < tol

    # ---------- Try to reduce rank by removing columns ----------
    def try_remove_column(U_mat, V_mat, W_mat, r, tol=1e-9):
        """Try to remove column r. Returns (success, new_U, new_V, new_W)."""
        rank = U_mat.shape[1]
        if rank <= 1:
            return False, U_mat, V_mat, W_mat

        T_r = tensor_of_column(U_mat[:, r], V_mat[:, r], W_mat[:, r])
        others = [c for c in range(rank) if c != r]
        M = np.column_stack([tensor_of_column(U_mat[:, c], V_mat[:, c], W_mat[:, c]) for c in others])

        try:
            coeffs, residuals, rank_m, sv = np.linalg.lstsq(M, T_r, rcond=None)
        except np.linalg.LinAlgError:
            return False, U_mat, V_mat, W_mat

        recon = M @ coeffs
        if np.max(np.abs(recon - T_r)) > tol:
            return False, U_mat, V_mat, W_mat

        new_U = U_mat.copy()
        new_V = V_mat.copy()
        new_W = W_mat.copy()
        for idx, c in enumerate(others):
            new_U[:, c] += coeffs[idx] * U_mat[:, r]
            new_V[:, c] += coeffs[idx] * V_mat[:, r]
            new_W[:, c] += coeffs[idx] * W_mat[:, r]

        new_U = np.delete(new_U, r, axis=1)
        new_V = np.delete(new_V, r, axis=1)
        new_W = np.delete(new_W, r, axis=1)

        if verify(new_U, new_V, new_W, tol):
            return True, new_U, new_V, new_W
        return False, U_mat, V_mat, W_mat

    # Iteratively remove columns
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for r in range(U.shape[1]):
            if time.time() > deadline - 10:
                break
            success, U, V, W = try_remove_column(U, V, W, r)
            if success:
                improved = True
                break

    # ---------- Try Strassen-based rank-42 decomposition ----------
    if U.shape[1] >= 42:
        U_s = np.zeros((n * m, 42))
        V_s = np.zeros((m * p, 42))
        W_s = np.zeros((p * n, 42))

        # Strassen 2x2x2 terms
        strassen_terms = [
            ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
            ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
            ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
            ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
            ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
            ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
            ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ]

        col_idx = 0
        for source in range(2):  # 2 column blocks of 2
            for c in range(3):    # 3 row blocks of 2
                for t in range(7): # 7 Strassen terms
                    u_factors, v_factors, w_factors = strassen_terms[t]
                    for (pi, pj), coeff in u_factors:
                        U_s[pi * 4 + (pj + 2 * source), col_idx] += coeff
                    for (qi, qj), coeff in v_factors:
                        V_s[(qi + 2 * source) * 6 + (2 * c + qj), col_idx] += coeff
                    for (ri, rj), coeff in w_factors:
                        W_s[(2 * c + rj) * 2 + ri, col_idx] += coeff
                    col_idx += 1

        if verify(U_s, V_s, W_s):
            U, V, W = U_s, V_s, W_s

    # ---------- Try to reduce the Strassen decomposition further ----------
    if U.shape[1] == 42:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for r in range(U.shape[1]):
                if time.time() > deadline - 10:
                    break
                success, U, V, W = try_remove_column(U, V, W, r)
                if success:
                    improved = True
                    break

    # ---------- Final verification ----------
    if not verify(U, V, W, tol=1e-9):
        raise ValueError("Decomposition verification failed")

    # Round to nearest half-integer for cleanliness
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2

    if verify(U_rounded, V_rounded, W_rounded, tol=1e-9):
        U, V, W = U_rounded, V_rounded, W_rounded

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
