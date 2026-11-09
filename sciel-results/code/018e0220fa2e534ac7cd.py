"""Matrix multiplication 2x3 x 3x4: rank-minimal exact tensor decomposition.

Strategy:
  1. Build a rank-22 Strassen-based block decomposition.
  2. Attempt rank reduction via:
     a. Merging proportional terms.
     b. Removing one term at a time (solving for the third factor).
     c. Removing two terms at a time.
     d. Iterated linear-combination elimination (ILP-style pruning).
  3. Final polish: round to nearest half-integer and re-verify.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_BUDGET = 138.0  # evaluator kills at 150s; keep 12s margin


def _tensor_index(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == T exactly (within tol)."""
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0.0)


def _build_system_for_W(U_r, V_r, T, n, m, p, nr):
    """Build linear system A * W_flat = T_flat to solve for W given U_r, V_r."""
    A_mat = np.zeros((n * m * p, p * nr), dtype=np.float64)
    b_vec = T.flatten()
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r_idx in range(nr):
                    A_mat[idx, k * nr + r_idx] = U_r[i, r_idx] * V_r[j, r_idx]
                idx += 1
    return A_mat, b_vec


def _build_system_for_U(U_r, V_r, W_r, T, n, m, p, nr):
    """Build linear system to solve for U given V_r and W_r."""
    A_mat = np.zeros((n * m * p, n * nr), dtype=np.float64)
    b_vec = T.flatten()
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r_idx in range(nr):
                    A_mat[idx, i * nr + r_idx] = V_r[j, r_idx] * W_r[k, r_idx]
                idx += 1
    return A_mat, b_vec


def _build_system_for_V(U_r, V_r, W_r, T, n, m, p, nr):
    """Build linear system to solve for V given U_r and W_r."""
    A_mat = np.zeros((n * m * p, m * nr), dtype=np.float64)
    b_vec = T.flatten()
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r_idx in range(nr):
                    A_mat[idx, j * nr + r_idx] = U_r[i, r_idx] * W_r[k, r_idx]
                idx += 1
    return A_mat, b_vec


def _check_solution(U_new, V_new, W_new, T, n, m, p):
    """Check if the new decomposition is valid."""
    reconstructed = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
    return np.allclose(reconstructed, T, atol=1e-9, rtol=0.0)


def _try_reduce(U, V, W, T, n, m, p):
    """Attempt to find a lower-rank decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]

    def _merge_proportional(U, V, W, rank):
        """Merge proportional terms iteratively."""
        merged = True
        while merged and rank > 1:
            merged = False
            for r in range(rank):
                for s in range(r + 1, rank):
                    if np.linalg.norm(U[:, r]) < 1e-12 or np.linalg.norm(U[:, s]) < 1e-12:
                        continue
                    if np.linalg.norm(V[:, r]) < 1e-12 or np.linalg.norm(V[:, s]) < 1e-12:
                        continue
                    ratio_u = np.dot(U[:, r], U[:, s]) / np.dot(U[:, s], U[:, s])
                    if (np.allclose(U[:, r], ratio_u * U[:, s], atol=1e-12) and
                            np.allclose(V[:, r], ratio_u * V[:, s], atol=1e-12) and
                            np.allclose(W[:, r], ratio_u * W[:, s], atol=1e-12)):
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        rank -= 1
                        merged = True
                        break
                if merged:
                    break
        return U, V, W, rank

    def _try_remove_one(U, V, W, T, n, m, p, rank):
        """Try removing one term, solving for W."""
        for r_to_remove in range(rank):
            if time.time() - START_TIME > TIME_BUDGET - 10:
                break
            mask = [i for i in range(rank) if i != r_to_remove]
            U_r = U[:, mask]
            V_r = V[:, mask]
            nr = len(mask)
            A_mat, b_vec = _build_system_for_W(U_r, V_r, T, n, m, p, nr)
            try:
                W_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_sol = W_sol.reshape(p, nr)
                if _check_solution(U_r, V_r, W_sol, T, n, m, p):
                    return U_r, V_r, W_sol, nr
            except np.linalg.LinAlgError:
                continue
        return None

    def _try_remove_two(U, V, W, T, n, m, p, rank):
        """Try removing two terms, solving for any one factor."""
        for r1 in range(rank):
            if time.time() - START_TIME > TIME_BUDGET - 10:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() - START_TIME > TIME_BUDGET - 10:
                    break
                mask = [i for i in range(rank) if i != r1 and i != r2]
                U_r = U[:, mask]
                V_r = V[:, mask]
                W_r = W[:, mask]
                nr = len(mask)

                # Solve for W
                A_mat, b_vec = _build_system_for_W(U_r, V_r, T, n, m, p, nr)
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_sol = W_sol.reshape(p, nr)
                    if _check_solution(U_r, V_r, W_sol, T, n, m, p):
                        return U_r, V_r, W_sol, nr
                except np.linalg.LinAlgError:
                    pass

                # Solve for U
                A_mat, b_vec = _build_system_for_U(U_r, V_r, W_r, T, n, m, p, nr)
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_sol = U_sol.reshape(n, nr)
                    if _check_solution(U_sol, V_r, W_r, T, n, m, p):
                        return U_sol, V_r, W_r, nr
                except np.linalg.LinAlgError:
                    pass

                # Solve for V
                A_mat, b_vec = _build_system_for_V(U_r, V_r, W_r, T, n, m, p, nr)
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_sol = V_sol.reshape(m, nr)
                    if _check_solution(U_r, V_sol, W_r, T, n, m, p):
                        return U_r, V_sol, W_r, nr
                except np.linalg.LinAlgError:
                    pass
        return None

    def _ilp_eliminate(U, V, W, rank):
        """Check if any term is a linear combination of others in all three factors."""
        eliminated = True
        while eliminated and rank > 1:
            eliminated = False
            for r_target in range(rank):
                if time.time() - START_TIME > TIME_BUDGET - 5:
                    break
                others = [i for i in range(rank) if i != r_target]
                U_others = U[:, others]
                V_others = V[:, others]
                W_others = W[:, others]
                try:
                    coeff_U, _, _, _ = np.linalg.lstsq(U_others, U[:, r_target], rcond=None)
                    if np.allclose(U_others @ coeff_U, U[:, r_target], atol=1e-10):
                        coeff_V, _, _, _ = np.linalg.lstsq(V_others, V[:, r_target], rcond=None)
                        if np.allclose(V_others @ coeff_V, V[:, r_target], atol=1e-10):
                            coeff_W, _, _, _ = np.linalg.lstsq(W_others, W[:, r_target], rcond=None)
                            if np.allclose(W_others @ coeff_W, W[:, r_target], atol=1e-10):
                                U = np.delete(U, r_target, axis=1)
                                V = np.delete(V, r_target, axis=1)
                                W = np.delete(W, r_target, axis=1)
                                rank -= 1
                                eliminated = True
                                break
                except np.linalg.LinAlgError:
                    pass
        return U, V, W, rank

    # Step 1: Merge proportional terms
    U, V, W, rank = _merge_proportional(U, V, W, rank)

    # Step 2: Iterative single- and double-term removal
    progress = True
    while progress and rank > 1:
        if time.time() - START_TIME > TIME_BUDGET - 10:
            break
        progress = False

        # Try removing one term
        result = _try_remove_one(U, V, W, T, n, m, p, rank)
        if result is not None:
            U, V, W, rank = result
            progress = True
            continue

        # Try removing two terms
        result = _try_remove_two(U, V, W, T, n, m, p, rank)
        if result is not None:
            U, V, W, rank = result
            progress = True
            continue

    # Step 3: ILP-style elimination
    U, V, W, rank = _ilp_eliminate(U, V, W, rank)

    # Step 4: Another round of merging + removal after ILP
    U, V, W, rank = _merge_proportional(U, V, W, rank)
    progress = True
    while progress and rank > 1:
        if time.time() - START_TIME > TIME_BUDGET - 5:
            break
        progress = False
        result = _try_remove_one(U, V, W, T, n, m, p, rank)
        if result is not None:
            U, V, W, rank = result
            progress = True
            continue
        result = _try_remove_two(U, V, W, T, n, m, p, rank)
        if result is not None:
            U, V, W, rank = result
            progress = True
            continue

    return U, V, W


def _build_rank22_decomposition(n, m, p):
    """Build the rank-22 Strassen-based decomposition for 2x3 x 3x4."""
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        """Add a Strassen rank-7 decomposition for a 2x2x2 block."""
        col = col_start
        # M1
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M2
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        # M3
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M4
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        # M5
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        # M6
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M7
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    # Block 1: A_L (2x2) * B_T1 (2x2) -> C[:, 0:2]
    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    c1 = {'00': 0, '01': 2, '10': 1, '11': 3}
    col = _add_strassen_block(a1, b1, c1, 0)

    # Block 2: A_L (2x2) * B_T2 (2x2) -> C[:, 2:4]
    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    col = _add_strassen_block(a1, b2, c2, col)

    # Block 3: A_R (2x1) * B_B (1x4) -> C (2x4), rank-8 schoolbook
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _tensor_index(n, m, p)

    # Start with rank-22 Strassen decomposition
    U, V, W = _build_rank22_decomposition(n, m, p)

    if not _verify(U, V, W, T):
        # Fallback: schoolbook rank n*m*p = 24
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction
    U, V, W = _try_reduce(U, V, W, T, n, m, p)

    # Verify after reduction
    if not _verify(U, V, W, T):
        # Revert to rank 22
        U, V, W = _build_rank22_decomposition(n, m, p)

    # Final polish: round to nearest half-integer
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2

    # Re-verify after rounding
    if _verify(U_rounded, V_rounded, W_rounded, T):
        U, V, W = U_rounded, V_rounded, W_rounded

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
    T = _tensor_index(2, 3, 4)
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify(U, V, W, T)}")
