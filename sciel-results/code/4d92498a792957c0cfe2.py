"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def _validate(U, V, W, T):
    n, m, p = T.shape
    rank = U.shape[1]
    recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                uj = U[i * m + j, r]
                if uj == 0:
                    continue
                for k in range(p):
                    recon[i, j, k] += uj * V[j * p + k, r] * W[k * n + i, r]
    return np.allclose(recon, T, atol=1e-9)

def _is_half_integer(arr, max_val=100):
    if not np.all(np.isfinite(arr)):
        return False
    doubled = arr * 2
    if not np.all(np.isclose(doubled, np.round(doubled), atol=1e-6)):
        return False
    if not np.all(np.abs(arr) <= max_val):
        return False
    return True

def _snap_half(arr):
    return np.round(arr * 2) / 2

def _build_strassen_block(i_off, j_off, k_off, U, V, W, col_start, m, n, p):
    """Add 7 Strassen terms for a 2x2x2 block at offsets (i_off, j_off, k_off)."""
    # Strassen multipliers for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    multipliers = [
        # (A_terms: [(ii,jj,coeff)], B_terms: [(jj,kk,coeff)])
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),   # M1
        ([((1,0,1),(1,1,1))], [((0,0,1))]),              # M2
        ([((0,0,1))], [((0,1,1),(1,1,-1))]),             # M3
        ([((1,1,1))], [((1,0,1),(0,0,-1))]),             # M4
        ([((0,0,1),(0,1,1))], [((1,1,1))]),              # M5
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))]),     # M6
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))]),     # M7
    ]

    # W contributions: (m_idx, (ii,kk), coeff)
    W_contribs = {
        0: [((0,0,1),(1,1,1))],     # M1 -> C00(+1), C11(+1)
        1: [((1,0,1),(1,1,-1))],    # M2 -> C10(+1), C11(-1)
        2: [((0,1,1),(1,1,1))],     # M3 -> C01(+1), C11(+1)
        3: [((0,0,1),(1,0,1))],     # M4 -> C00(+1), C10(+1)
        4: [((0,0,-1),(0,1,1))],    # M5 -> C00(-1), C01(+1)
        5: [((1,1,1))],             # M6 -> C11(+1)
        6: [((0,0,1))],             # M7 -> C00(+1)
    }

    for mi, (a_terms, b_terms) in enumerate(multipliers):
        col = col_start + mi
        # Fill U
        for (ii, jj, coeff) in a_terms:
            i_g = i_off + ii
            j_g = j_off + jj
            U[i_g * m + j_g, col] += coeff
        # Fill V
        for (jj, kk, coeff) in b_terms:
            j_g = j_off + jj
            k_g = k_off + kk
            V[j_g * p + k_g, col] += coeff
        # Fill W
        for (ii, kk, coeff) in W_contribs[mi]:
            i_g = i_off + ii
            k_g = k_off + kk
            W[k_g * n + i_g, col] += coeff

    return col_start + 7

def _build_decomposition():
    """Build the direct-sum decomposition: 1x4x5 (rank 4) + 4 Strassen 2x2x2 blocks (rank 28)."""
    n, m, p = DIMENSIONS  # 3, 4, 5
    total_rank = 32
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0

    # Part 1: 1x4x5 block (i=0, all j, all k) -- rank 4
    # For each (j,k), one term: U[0*m+j, col]=1, V[j*p+k, col]=1, W[k*n+0, col]=1
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    # col is now 20

    # Part 2: 2x4x5 blocks (i=1,2; all j; all k)
    # Split into 4 Strassen 2x2x2 blocks
    # Block A: i_off=1, j_off=0, k_off=0
    col = _build_strassen_block(1, 0, 0, U, V, W, col, m, n, p)
    # Block B: i_off=1, j_off=2, k_off=0
    col = _build_strassen_block(1, 2, 0, U, V, W, col, m, n, p)
    # Block C: i_off=1, j_off=0, k_off=2
    col = _build_strassen_block(1, 0, 2, U, V, W, col, m, n, p)
    # Block D: i_off=1, j_off=2, k_off=2
    col = _build_strassen_block(1, 2, 2, U, V, W, col, m, n, p)

    assert col == total_rank, f"Expected {total_rank}, got {col}"
    return U, V, W

def _try_remove_column(U, V, W, T, r_remove):
    """Try to remove column r_remove by solving for adjustments to remaining columns."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    if R <= 1:
        return None

    mask = [i for i in range(R) if i != r_remove]
    U_rem = U[:, mask].copy()
    V_rem = V[:, mask].copy()
    W_rem = W[:, mask].copy()

    # Compute residual from removed column
    residual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            uj = U[i * m + j, r_remove]
            if uj != 0:
                for k in range(p):
                    residual[i, j, k] = uj * V[j * p + k, r_remove] * W[k * n + i, r_remove]

    # Try to absorb residual into U (keep V, W fixed)
    # For each (i,j): sum_r delta_U[i*m+j, r] * V_rem[j*p+k, r] * W_rem[k*n+i, r] = residual[i,j,k]
    delta_U = np.zeros((n * m, len(mask)))
    solvable = True
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, len(mask)))
            for k in range(p):
                M[k, :] = V_rem[j * p + k, :] * W_rem[k * n + i, :]
            try:
                sol, res, rank_s, sv = np.linalg.lstsq(M.T, residual[i, j, :], rcond=None)
                delta_U[i * m + j, :] = sol
            except:
                solvable = False
                break
        if not solvable:
            break

    if not solvable:
        return None

    U_new = U_rem + delta_U

    # Check half-integer constraint
    if not _is_half_integer(U_new, max_val=1000):
        return None

    # Validate
    if not _validate(U_new, V_rem, W_rem, T):
        return None

    return U_new, V_rem, W_rem

def _try_reduce(U, V, W, T, max_time, rng):
    """Iteratively try to remove columns."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    start = time.time()

    while R > 1:
        if time.time() - start > max_time:
            break

        found = False
        # Try removing each column
        for r_remove in range(R):
            if time.time() - start > max_time:
                break

            result = _try_remove_column(U, V, W, T, r_remove)
            if result is not None:
                U, V, W = result
                R = U.shape[1]
                found = True
                break

        if not found:
            break

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))

    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Build the direct-sum decomposition (rank 32)
    U, V, W = _build_decomposition()

    # Validate
    if not _validate(U, V, W, T):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n * m)]
        V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m * p)]
        W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to reduce rank further
    if time.time() < deadline - 20:
        rng = random.Random(42)
        U_red, V_red, W_red = _try_reduce(U.copy(), V.copy(), W.copy(), T, max_time=30, rng=rng)
        R_final = U_red.shape[1]
        if R_final < best_rank:
            best_rank = R_final
            best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()

    # Try random perturbations and re-reduction
    if time.time() < deadline - 10:
        rng = random.Random(123)
        np.random.seed(123)
        for attempt in range(20):
            if time.time() > deadline - 15:
                break

            # Randomly swap/scale some columns to explore equivalent decompositions
            U_try = best_U.copy()
            V_try = best_V.copy()
            W_try = best_W.copy()

            # Random column permutation
            perm = list(range(best_rank))
            rng.shuffle(perm)
            U_try = U_try[:, perm]
            V_try = V_try[:, perm]
            W_try = W_try[:, perm]

            # Try reduction
            U_red, V_red, W_red = _try_reduce(U_try, V_try, W_try, T, max_time=5, rng=rng)
            R_final = U_red.shape[1]
            if R_final < best_rank:
                best_rank = R_final
                best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()

    # Convert to lists
    U_list = [[float(best_U[i, r]) for r in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, r]) for r in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, r]) for r in range(best_rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
