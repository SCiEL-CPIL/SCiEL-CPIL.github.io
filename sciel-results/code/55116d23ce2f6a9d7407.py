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
    # T[i, j, k] = 1 when A[i,j]*B[j,k] contributes to C[i,k]
    # U[i, r], V[j, r], W[k, r] with i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # Schoolbook: for each j in 0..m-1, for each (i, k): one term
    # Actually schoolbook rank = n * m * p = 48
    # Term r: U[i, r] = 1 if i//m == i_A, V[j, r] = 1 if j//p == j_B, W[k, r] = 1 if k%n == i_A
    # where j_A == j_B

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

    # ---------- Rank reduction via tensor-column dependency detection ----------
    # Two columns r1, r2 of (U,V,W) are "redundant" if there exist scalars a, b (not both 0)
    # such that a*U[:,r1]+b*U[:,r2], a*V[:,r1]+b*V[:,r2], a*W[:,r1]+b*W[:,r2]
    # produces a zero tensor contribution. For rank-1 tensors, column r defines a rank-1 tensor
    # T_r[i,j,k] = U[i,r]*V[j,r]*W[k,r]. The full tensor is sum_r T_r.
    # A set of columns is a valid decomposition if sum_r T_r = T_target.
    # To reduce rank: find a column r that is a linear combination (in the tensor space) of others.
    # T_r = sum_{r' != r} c_{r,r'} T_{r'}
    # This is a linear system in the tensor entries.

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

    # Attempt to reduce rank by removing one column at a time.
    # For each candidate column r, check if it can be expressed as a linear combination
    # of the remaining columns in the tensor space.
    # We solve: T_r = sum_{r' in remaining} c_{r'} T_{r'}
    # This is a least-squares problem in the tensor space (dimension n*m*p*n*p = 8*24*12 = 2304).

    def try_remove_column(U_mat, V_mat, W_mat, r, tol=1e-9):
        """Try to remove column r. Returns (success, new_U, new_V, new_W)."""
        rank = U_mat.shape[1]
        if rank <= 1:
            return False, U_mat, V_mat, W_mat

        # Build tensor vectors for all columns
        T_r = tensor_of_column(U_mat[:, r], V_mat[:, r], W_mat[:, r])

        # Build matrix of other columns' tensor vectors
        others = [c for c in range(rank) if c != r]
        M = np.column_stack([tensor_of_column(U_mat[:, c], V_mat[:, c], W_mat[:, c]) for c in others])

        # Solve M @ coeffs = T_r (least squares)
        try:
            coeffs, residuals, rank_m, sv = np.linalg.lstsq(M, T_r, rcond=None)
        except np.linalg.LinAlgError:
            return False, U_mat, V_mat, W_mat

        # Check residual
        recon = M @ coeffs
        if np.max(np.abs(recon - T_r)) > tol:
            return False, U_mat, V_mat, W_mat

        # Remove column r and adjust the others
        new_U = U_mat.copy()
        new_V = V_mat.copy()
        new_W = W_mat.copy()
        for idx, c in enumerate(others):
            new_U[:, c] += coeffs[idx] * U_mat[:, r]
            new_V[:, c] += coeffs[idx] * V_mat[:, r]
            new_W[:, c] += coeffs[idx] * W_mat[:, r]

        # Drop column r
        new_U = np.delete(new_U, r, axis=1)
        new_V = np.delete(new_V, r, axis=1)
        new_W = np.delete(new_W, r, axis=1)

        # Verify the full tensor is still correct
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
                break  # restart scan from 0 after a removal

    # ---------- Try Strassen-based rank-42 as a fallback/baseline ----------
    # If schoolbook reduction didn't get below 42, use the Strassen construction.
    if U.shape[1] >= 42:
        # Build Strassen rank-42 decomposition
        U_s = np.zeros((n * m, 42))
        V_s = np.zeros((m * p, 42))
        W_s = np.zeros((p * n, 42))

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
        for source in range(2):
            for c in range(3):
                for t in range(7):
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

    # ---------- Final verification ----------
    if not verify(U, V, W, tol=1e-9):
        raise ValueError("Decomposition verification failed")

    # Round to nearest half-integer for cleanliness
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    if not verify(U, V, W, tol=1e-9):
        # If rounding broke it, use unrounded
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        r = 0
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    U[i_A * m + j, r] = 1.0
                    V[j * p + k_B, r] = 1.0
                    W[k_B * n + i_A, r] = 1.0
                    r += 1

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
