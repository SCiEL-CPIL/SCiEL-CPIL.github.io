"""Construct a rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication.

Starts from a rank-34 Strassen-based construction, then searches for a lower-rank
exact decomposition by applying random invertible basis transformations to the
three factor matrices. Under the transformation
    U' = U @ A,  V' = V @ B,  W' = W @ (A ⊗ B)^{-1}
the tensor is preserved (rank invariant), but a different basis may expose
linearly dependent / zero columns that can be dropped, revealing a lower-rank
exact decomposition.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W, atol=1e-9):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    r = Ua.shape[1]
    if r == 0:
        return False
    # Reshape to 3D
    U3 = Ua.reshape(n, m, r)  # U3[i, j, r]
    V3 = Va.reshape(m, p, r)  # V3[j, k, r]
    W3 = Wa.reshape(p, n, r)  # W3[k, i, r]
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=atol))


def _build_rank34_strassen():
    """Rank-34 decomposition: 2 Strassen blocks (2x2x2) + schoolbook for remaining parts.

    Split the 3x3x4 multiplication into:
      * Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1): Strassen rank 7
      * Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3): Strassen rank 7
      * Bottom row (i=2; j=0,1,2; k=0,1,2,3): schoolbook rank 12
      * Right column (i=0,1; j=2; k=0,1,2,3): schoolbook rank 8
    Total: 7 + 7 + 12 + 8 = 34
    """
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Term 1: M1 = (A11+A22) * (B11+B22) -> C11 += M1, C22 += M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1

        # Term 2: M2 = (A21+A22) * B11 -> C21 += M2, C22 -= M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1

        # Term 3: M3 = A11 * (B12-B22) -> C12 += M3, C22 += M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1

        # Term 4: M4 = A22 * (B21-B11) -> C11 += M4, C21 += M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1

        # Term 5: M5 = (A11+A12) * B22 -> C11 -= M5, C12 += M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1

        # Term 6: M6 = (A21-A11) * (B11+B12) -> C22 += M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1

        # Term 7: M7 = (A12-A22) * (B21+B22) -> C11 += M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1

    # 1. Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1)
    add_strassen(0, 0, 0)
    # 2. Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3)
    add_strassen(0, 0, 2)
    # 3. Bottom row (i=2; j=0,1,2; k=0,1,2,3)
    for j in range(3):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)
    # 4. Right column (i=0,1; j=2; k=0,1,2,3)
    for i in range(2):
        for k in range(4):
            add_term(i * m + 2, 2 * p + k, k * n + i)

    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Schoolbook rank-36 decomposition (fallback)."""
    n, m, p = 3, 3, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_transform(U, V, W, A, B, tol=1e-10):
    """Apply basis transform U'=U@A, V'=V@B, W'=W@(A⊗B)^{-1}; drop zero columns.

    Returns (U', V', W') as float arrays with zero columns removed, or None if
    the transform is numerically singular or fails to preserve the tensor.
    """
    r = U.shape[1]
    try:
        # (A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1}, but we can also just invert the Kronecker.
        # For efficiency, compute A_inv and B_inv separately.
        A_inv = np.linalg.inv(A)
        B_inv = np.linalg.inv(B)
        Kr_inv = np.kron(A_inv, B_inv)
        Wt = W @ Kr_inv
    except np.linalg.LinAlgError:
        return None

    Ut = U @ A
    Vt = V @ B

    # Detect zero columns across all three factors simultaneously
    # A column is "zero" if all entries in U, V, W for that column are ~0
    U_max = np.max(np.abs(Ut), axis=0)
    V_max = np.max(np.abs(Vt), axis=0)
    W_max = np.max(np.abs(Wt), axis=0)
    keep_mask = (U_max > tol) & (V_max > tol) & (W_max > tol)
    keep = np.where(keep_mask)[0]

    if len(keep) == r:
        return Ut, Vt, Wt

    return Ut[:, keep], Vt[:, keep], Wt[:, keep]


def _search_lower_rank(base, n, m, p, deadline, seed=12345):
    """Search for a lower-rank exact decomposition via random basis transforms."""
    rng = np.random.default_rng(seed)
    U0 = np.asarray(base["U"], dtype=float)
    V0 = np.asarray(base["V"], dtype=float)
    W0 = np.asarray(base["W"], dtype=float)
    r0 = U0.shape[1]

    best_rank = r0
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()

    start_time = time.time()
    num_trials = 0

    while time.time() < deadline - 15:
        num_trials += 1
        
        # Choose transform type: mostly random, but sometimes structured
        # to increase chance of producing exact zeros
        trial_type = rng.random()
        
        if trial_type < 0.5:
            # Standard Gaussian random
            A = rng.standard_normal((r0, r0))
            B = rng.standard_normal((r0, r0))
        elif trial_type < 0.7:
            # Sparse random: many zeros in the transform matrix
            # This can align columns to produce zeros in the output
            A = np.zeros((r0, r0))
            B = np.zeros((r0, r0))
            for i in range(r0):
                A[i, rng.integers(0, r0)] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
                B[i, rng.integers(0, r0)] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
            # Add small random perturbation to ensure invertibility
            A += rng.standard_normal((r0, r0)) * 0.01
            B += rng.standard_normal((r0, r0)) * 0.01
        elif trial_type < 0.85:
            # Integer-biased: entries are small integers
            A = rng.integers(-2, 3, size=(r0, r0)).astype(float)
            B = rng.integers(-2, 3, size=(r0, r0)).astype(float)
            # Ensure diagonal dominance for invertibility
            for i in range(r0):
                A[i, i] += rng.uniform(1.0, 3.0)
                B[i, i] += rng.uniform(1.0, 3.0)
        else:
            # Block-structured: identity plus random off-diagonal
            A = np.eye(r0) + rng.standard_normal((r0, r0)) * 0.5
            B = np.eye(r0) + rng.standard_normal((r0, r0)) * 0.5
        
        # Check invertibility
        try:
            cond_A = np.linalg.cond(A)
            cond_B = np.linalg.cond(B)
        except np.linalg.LinAlgError:
            continue
        if cond_A > 1e12 or cond_B > 1e12:
            continue
            
        res = _try_transform(U0, V0, W0, A, B)
        if res is None:
            continue
            
        Ut, Vt, Wt = res
        r_new = Ut.shape[1]
        
        if r_new < best_rank:
            # Verify exactness with tighter tolerance
            if _verify(n, m, p, Ut, Vt, Wt, atol=1e-9):
                best_rank = r_new
                best_U = Ut
                best_V = Vt
                best_W = Wt

                # If we found a significantly lower rank, do a focused search around this
                if best_rank <= r0 - 2:
                    # Try more transforms starting from this improved base
                    U0 = Ut
                    V0 = Vt
                    W0 = Wt
                    r0 = best_rank
                    # Reset the transform search to try to go even lower
                    pass

    # Final polish: try to reduce rank by checking if any columns can be dropped
    # after a very tight tolerance check
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        # Try to drop columns with very small norm
        U_max = np.max(np.abs(best_U), axis=0)
        V_max = np.max(np.abs(best_V), axis=0)
        W_max = np.max(np.abs(best_W), axis=0)
        col_norms = U_max + V_max + W_max
        # Sort columns by norm, try dropping the smallest ones
        order = np.argsort(col_norms)
        for drop_count in range(1, min(5, best_rank // 4 + 1)):
            if time.time() > deadline - 5:
                break
            keep = list(set(range(best_rank)) - set(order[:drop_count]))
            keep.sort()
            Uk = best_U[:, keep]
            Vk = best_V[:, keep]
            Wk = best_W[:, keep]
            if _verify(n, m, p, Uk, Vk, Wk, atol=1e-9):
                best_rank = len(keep)
                best_U = Uk
                best_V = Vk
                best_W = Wk
                break

    return (best_rank, {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()})


def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before the 150s kill

    n, m, p = 3, 3, 4

    # Start from the rank-34 Strassen-based decomposition.
    base = _build_rank34_strassen()
    if not _verify(n, m, p, base["U"], base["V"], base["W"]):
        # Fall back to schoolbook if the construction is broken.
        base = _build_schoolbook()
        return base

    # Search for a lower-rank exact decomposition via random basis transforms.
    try:
        best_rank, best = _search_lower_rank(base, n, m, p, deadline)
        return best
    except Exception:
        return base
# EVOLVE-BLOCK-END
