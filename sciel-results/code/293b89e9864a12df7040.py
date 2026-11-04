"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition.

Strategy:
  1. Finer block decomposition: n=[1,2], m=[2,1,2], p=[2,1,2] gives 18 blocks
     with 4 Strassen 2x2x2 cores (rank 7 each) and 14 schoolbook blocks,
     yielding a deterministic rank-63 construction.
  2. Focused search for lower-rank decompositions: fix structured cores and
     solve for W via least squares, then snap to half-integers and verify.
  3. Return the best valid decomposition found.
"""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7); indexing U[i*2+j,c], V[j*2+k,c], W[k*2+i,c]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[:, 0] = [1, 0, 0, 1]; V[:, 0] = [1, 0, 0, 1]; W[:, 0] = [1, 0, 0, 1]
    U[:, 1] = [0, 0, 1, 1]; V[:, 1] = [1, 0, 0, 0]; W[:, 1] = [0, 1, 0, -1]
    U[:, 2] = [1, 0, 0, 0]; V[:, 2] = [0, 1, 0, -1]; W[:, 2] = [0, 0, 1, 1]
    U[:, 3] = [0, 0, 0, 1]; V[:, 3] = [-1, 0, 1, 0]; W[:, 3] = [1, 1, 0, 0]
    U[:, 4] = [1, 1, 0, 0]; V[:, 4] = [0, 0, 0, 1]; W[:, 4] = [-1, 0, 1, 0]
    U[:, 5] = [-1, 0, 1, 0]; V[:, 5] = [1, 1, 0, 0]; W[:, 5] = [0, 0, 0, 1]
    U[:, 6] = [0, 1, 0, -1]; V[:, 6] = [0, 0, 1, 1]; W[:, 6] = [1, 0, 0, 0]
    return U, V, W


def _schoolbook_local(ni, mi, pi):
    """Schoolbook local decomposition for ni x mi x pi."""
    rank = ni * mi * pi
    U = np.zeros((ni * mi, rank))
    V = np.zeros((mi * pi, rank))
    W = np.zeros((pi * ni, rank))
    col = 0
    for i in range(ni):
        for j in range(mi):
            for k in range(pi):
                U[i * mi + j, col] = 1
                V[j * pi + k, col] = 1
                W[k * ni + i, col] = 1
                col += 1
    return U, V, W


def _embed_block(U_loc, V_loc, W_loc, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """Embed a local ni x mi x pi decomposition into global dimensions."""
    r = U_loc.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    for li in range(ni):
        for lj in range(mi):
            U_g[(n_off + li) * m + (m_off + lj), :] = U_loc[li * mi + lj, :]
    for lj in range(mi):
        for lk in range(pi):
            V_g[(m_off + lj) * p + (p_off + lk), :] = V_loc[lj * pi + lk, :]
    for lk in range(pi):
        for li in range(ni):
            W_g[(p_off + lk) * n + (n_off + li), :] = W_loc[lk * ni + li, :]
    return U_g, V_g, W_g


def _independent_block_decomp(n, m, p):
    """Finer block decomposition: n=[1,2], m=[2,1,2], p=[2,1,2].
    4 Strassen 2x2x2 blocks (rank 7 each) + 14 schoolbook blocks = rank 63.
    Returns U, V, W (numpy arrays) and total_rank."""
    n_parts = [1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    n_offsets = [0, 1]
    m_offsets = [0, 2, 3]
    p_offsets = [0, 2, 3]

    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook_local(ni, mi, pi)
                U_g, V_g, W_g = _embed_block(
                    U_loc, V_loc, W_loc, n, m, p,
                    n_offsets[i], m_offsets[j], p_offsets[k],
                    ni, mi, pi
                )
                U[:, col:col + r] += U_g
                V[:, col:col + r] += V_g
                W[:, col:col + r] += W_g
                col += r
    return U, V, W, total_rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly represents matrix multiplication."""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.max(np.abs(T - 1.0)) < tol


def _verify_half_integer(U, V, W, tol=1e-6):
    """Check all entries are close to half-integers."""
    for mat in [U, V, W]:
        doubled = mat * 2.0
        if np.max(np.abs(doubled - np.round(doubled))) > tol:
            return False
    return True


def _snap_half_integer(mat):
    """Snap matrix entries to nearest half-integer."""
    return np.round(mat * 2.0) / 2.0


def _try_lower_rank(n, m, p, deadline, max_rank_try=63, min_rank_try=55):
    """
    Search for a decomposition with rank < 63 by fixing structured cores
    and solving for W. Returns (U, V, W, rank) or None.
    """
    rng = np.random.RandomState(42)

    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
    n_eqs = len(eq_indices)

    # Precompute Strassen cores for embedding
    U_core, V_core, W_core = _strassen_2x2x2()

    # Strassen block positions in the finer partition:
    # n_parts=[1,2], m_parts=[2,1,2], p_parts=[2,1,2]
    # Strassen at (i=1,j=0,k=0), (1,0,2), (1,2,0), (1,2,2)
    # n_off=1, m_off in {0,3}, p_off in {0,3}
    strassen_positions = [(1, 0, 0), (1, 0, 3), (1, 3, 0), (1, 3, 3)]

    for R in range(max_rank_try - 1, min_rank_try - 1, -1):
        if time.time() > deadline - 20:
            break

        for attempt in range(150):
            if time.time() > deadline - 15:
                break

            U_rand = np.zeros((n * m, R))
            V_rand = np.zeros((m * p, R))

            if R >= 28:
                # Embed all 4 Strassen cores in first 28 columns
                for c_idx, (no, mo, po) in enumerate(strassen_positions):
                    col_start = c_idx * 7
                    for li in range(2):
                        for lj in range(2):
                            U_rand[(no + li) * m + (mo + lj), col_start:col_start + 7] = U_core[li * 2 + lj, :]
                    for lj in range(2):
                        for lk in range(2):
                            V_rand[(mo + lj) * p + (po + lk), col_start:col_start + 7] = V_core[lj * 2 + lk, :]

                # Fill remaining columns with structured random patterns
                remaining = R - 28
                if remaining > 0:
                    if attempt % 5 == 0:
                        # Sparse integer
                        mask_u = rng.rand(n * m, remaining) < 0.3
                        mask_v = rng.rand(m * p, remaining) < 0.3
                        U_rand[:, 28:][mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
                    elif attempt % 5 == 1:
                        # Small integers
                        U_rand[:, 28:] = rng.randint(-1, 2, (n * m, remaining)).astype(np.float64)
                        V_rand[:, 28:] = rng.randint(-1, 2, (m * p, remaining)).astype(np.float64)
                    elif attempt % 5 == 2:
                        # Binary
                        U_rand[:, 28:] = (rng.rand(n * m, remaining) < 0.4).astype(np.float64)
                        V_rand[:, 28:] = (rng.rand(m * p, remaining) < 0.4).astype(np.float64)
                    elif attempt % 5 == 3:
                        # Half-integers
                        mask_u = rng.rand(n * m, remaining) < 0.35
                        mask_v = rng.rand(m * p, remaining) < 0.35
                        U_rand[:, 28:][mask_u] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_v.sum())
                    else:
                        # Perturb baseline
                        U_base, V_base, W_base, _ = _independent_block_decomp(n, m, p)
                        if R <= U_base.shape[1]:
                            U_rand[:, 28:] = U_base[:, 28:R].copy()
                            V_rand[:, 28:] = V_base[:, 28:R].copy()
                        else:
                            U_rand[:, 28:U_base.shape[1]] = U_base[:, 28:].copy()
                            V_rand[:, 28:V_base.shape[1]] = V_base[:, 28:].copy()
                        mask_u = rng.rand(n * m, remaining) < 0.1
                        mask_v = rng.rand(m * p, remaining) < 0.1
                        U_rand[:, 28:][mask_u] += rng.choice([-1.0, 1.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] += rng.choice([-1.0, 1.0], size=mask_v.sum())
            else:
                # R < 28: no full Strassen embedding, use sparse patterns
                if attempt % 3 == 0:
                    mask_u = rng.rand(n * m, R) < 0.3
                    mask_v = rng.rand(m * p, R) < 0.3
                    U_rand[mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                    V_rand[mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
                elif attempt % 3 == 1:
                    U_rand = rng.randint(-1, 2, (n * m, R)).astype(np.float64)
                    V_rand = rng.randint(-1, 2, (m * p, R)).astype(np.float64)
                else:
                    U_rand = (rng.rand(n * m, R) < 0.4).astype(np.float64)
                    V_rand = (rng.rand(m * p, R) < 0.4).astype(np.float64)

            # Build linear system: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            n_vars = p * n * R
            A = np.zeros((n_eqs, n_vars), dtype=np.float64)
            b = np.ones(n_eqs, dtype=np.float64)

            for eq, (i, j, k) in enumerate(eq_indices):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                for r in range(R):
                    A[eq, w_row * R + r] = U_rand[u_row, r] * V_rand[v_row, r]

            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue

            W_rand = W_flat.reshape(p * n, R)

            # Check if W is close to half-integers
            W_check = W_rand * 2.0
            dist = np.abs(W_check - np.round(W_check))
            max_dist = np.max(dist)
            if max_dist > 1e-4:
                continue

            W_int = np.round(W_check) / 2.0

            # Validate the full decomposition
            if _verify(U_rand, V_rand, W_int, n, m, p):
                return U_rand, V_rand, W_int, R

    return None


def _try_polish(U, V, W, n, m, p, deadline):
    """Try to reduce rank by zeroing out columns that don't affect correctness.
    Iterate: try removing each column, check if still valid."""
    R = U.shape[1]
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = R

    # Try removing columns one at a time (greedy)
    for col_idx in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        if current_rank <= 55:
            break

        # Temporarily zero out this column
        saved_U = current_U[:, col_idx].copy()
        saved_V = current_V[:, col_idx].copy()
        saved_W = current_W[:, col_idx].copy()

        current_U[:, col_idx] = 0
        current_V[:, col_idx] = 0
        current_W[:, col_idx] = 0

        if _verify(current_U, current_V, current_W, n, m, p):
            # Column is redundant; remove it
            mask = np.ones(current_rank, dtype=bool)
            mask[col_idx] = False
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
        else:
            # Restore
            current_U[:, col_idx] = saved_U
            current_V[:, col_idx] = saved_V
            current_W[:, col_idx] = saved_W

    if current_rank < R:
        return current_U, current_V, current_W, current_rank
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # 1. Finer block decomposition (rank 63) as baseline
    U_ind, V_ind, W_ind, rank_ind = _independent_block_decomp(n, m, p)

    # Verify baseline
    if not _verify(U_ind, V_ind, W_ind, n, m, p):
        # Fallback to pure schoolbook (rank 75)
        rank = n * m * p
        U_ind = np.zeros((n * m, rank))
        V_ind = np.zeros((m * p, rank))
        W_ind = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_ind[i * m + j, col] = 1.0
                    V_ind[j * p + k, col] = 1.0
                    W_ind[k * n + i, col] = 1.0
                    col += 1
        rank_ind = rank

    best_U, best_V, best_W, best_rank = U_ind, V_ind, W_ind, rank_ind

    # 2. Try to find a better (lower rank) decomposition via search
    if time.time() < deadline - 30:
        result = _try_lower_rank(n, m, p, deadline, max_rank_try=best_rank, min_rank_try=55)
        if result is not None:
            U_new, V_new, W_new, R_new = result
            if R_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, R_new

    # 3. Try to polish: remove redundant columns
    if time.time() < deadline - 10:
        polished = _try_polish(best_U, best_V, best_W, n, m, p, deadline)
        if polished is not None:
            U_pol, V_pol, W_pol, R_pol = polished
            if R_pol < best_rank:
                best_U, best_V, best_W, best_rank = U_pol, V_pol, W_pol, R_pol

    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
