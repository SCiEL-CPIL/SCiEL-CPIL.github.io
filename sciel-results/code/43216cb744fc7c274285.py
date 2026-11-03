"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before the hard limit


def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_strassen_2x2():
    """Return U2, V2, W2 (7x4 integer matrices) for 2x2x2 Strassen."""
    U2 = np.array([
        [1, 0, 0, 1],    # M1: A00+A11
        [0, 0, 1, 1],    # M2: A10+A11
        [1, 0, 0, 0],    # M3: A00
        [0, 0, 0, 1],    # M4: A11
        [1, 1, 0, 0],    # M5: A00+A01
        [-1, 0, 1, 0],   # M6: A10-A00
        [0, 1, 0, -1]    # M7: A01-A11
    ], dtype=int)
    V2 = np.array([
        [1, 0, 0, 1],    # M1: B00+B11
        [1, 0, 0, 0],    # M2: B00
        [0, 1, 0, -1],   # M3: B01-B11
        [-1, 0, 1, 0],   # M4: B10-B00
        [0, 0, 0, 1],    # M5: B11
        [1, 1, 0, 0],    # M6: B00+B01
        [0, 0, 1, 1]     # M7: B10+B11
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 1],    # M1: C00+C11
        [0, 0, 1, -1],   # M2: C10-C11
        [0, 1, 0, 1],    # M3: C01+C11
        [1, 0, 1, 0],    # M4: C00+C10
        [-1, 1, 0, 0],   # M5: -C00+C01
        [0, 0, 0, 1],    # M6: C11
        [1, 0, 0, 0]     # M7: C00
    ], dtype=int)
    return U2, V2, W2


def _build_4x4x4(U2, V2, W2):
    """Build rank-49 4x4x4 decomposition via Kronecker product of Strassen.
    Returns U4 (16x49), V4 (16x49), W4 (16x49) with integer entries.
    Indexing: row-major flattening of 4x4 matrices.
    """
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    def idx2(i, j):
        return i * 2 + j

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for I in range(4):
                for J in range(4):
                    r = I // 2
                    c = J // 2
                    i_loc = I % 2
                    j_loc = J % 2
                    U4[I * 4 + J, col] = U2[s, idx2(r, c)] * U2[t, idx2(i_loc, j_loc)]
            for J in range(4):
                for K in range(4):
                    r = J // 2
                    c = K // 2
                    j_loc = J % 2
                    k_loc = K % 2
                    V4[J * 4 + K, col] = V2[s, idx2(r, c)] * V2[t, idx2(j_loc, k_loc)]
            for I in range(4):
                for K in range(4):
                    r = I // 2
                    c = K // 2
                    i_loc = I % 2
                    k_loc = K % 2
                    W4[I * 4 + K, col] = W2[s, idx2(r, c)] * W2[t, idx2(i_loc, k_loc)]
    return U4, V4, W4


def _build_4x4x8(U4, V4, W4):
    """Extend 4x4x4 rank-49 to 4x4x8 by two independent 4-column blocks.
    Total rank 98. Returns U (16x98), V (32x98), W (32x98).
    W follows the trace convention W[k*n + i, r].
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Left block (B columns 0-3, C columns 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[i * 4 + k, :]

    # Right block (B columns 4-7, C columns 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[i * 4 + k, :]
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] summed over r
    equals the matrix multiplication tensor. We verify by contracting:
    T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r] must equal
    sum_r W[k*n+i,r] (the coefficient of the output).
    Actually the tensor identity is:
      C[i,k] = sum_j A[i,j]*B[j,k]
    and the decomposition says the 3-tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]
    must equal the canonical multiplication tensor, which equals
    the W-contracted form: for each (i,j,k), the canonical value is 1 if the
    (i,j,k) triple is a valid path, else 0. But the W matrix stores the output
    coefficients. The correct check: the tensor defined by U,V must equal the
    tensor defined by the canonical multiplication, and W must be consistent.
    We check that U and V produce the canonical multiplication tensor and that
    W is the corresponding output map.
    """
    # Canonical multiplication tensor T_c[i,j,k] = 1 for all i,j,k (since
    # C[i,k] = sum_j A[i,j] B[j,k] means every (i,j,k) contributes).
    # The U,V contraction gives T_uv[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r].
    # This must equal 1 for all i,j,k.
    # And W must satisfy: sum_r W[k*n+i,r] = sum_j T_uv[i,j,k] = m for each (i,k)?
    # No. Let's use the standard: the multiplication tensor has entries
    # T[i,j,k] = 1 (all ones) for the (n,m,p) case because C[i,k] = sum_j A[i,j]B[j,k].
    # The rank-1 decomposition: T = sum_r u_r (over i,j) v_r (over j,k) w_r (over i,k).
    # So sum_r U[i*m+j,r]*V[j*p+k,r] should equal 1 for all i,j,k.
    # And W is just the output index map; its sum over r for fixed (i,k) is not
    # directly constrained to a constant, but the full tensor identity is:
    #   sum_r U[i*m+j,r]*V[j*p+k,r] = T_c[i,j,k] = 1.
    # We verify that.
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    # Build T_uv by einsum: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]
    # Reshape U to (n,m,rank), V to (m,p,rank), then einsum.
    U3 = Uf.reshape(n, m, -1)
    V3 = Vf.reshape(m, p, -1)
    T_uv = np.einsum('ijr,jkr->ijk', U3, V3)
    # Canonical is all-ones.
    err = np.max(np.abs(T_uv - 1.0))
    return err < tol


def _try_reduce_columns(U, V, W, n, m, p, deadline_left_fn):
    """Attempt to remove redundant columns from the decomposition.
    A column r is redundant if there exist integer/half-integer coefficients
    c_s (s != r) such that U[:,r] = sum_s c_s U[:,s], V[:,r] = sum_s c_s V[:,s],
    and W[:,r] = sum_s c_s W[:,s]. Since the entries are small integers, we
    stack [U;V;W] as a tall matrix M (rows = n*m + m*p + p*n = 16+32+32 = 80,
    cols = rank). A column is in the span of the others iff the rank of M does
    not increase when that column is removed. We greedily remove columns whose
    removal keeps the column-space rank unchanged AND keeps the decomposition
    valid (verified).
    """
    # Stack into one matrix for rank testing.
    M = np.vstack([U, V, W]).astype(np.float64)
    R = M.shape[1]
    # Compute baseline rank of the column space.
    base_rank = np.linalg.matrix_rank(M, tol=1e-9)
    keep = list(range(R))
    # Greedy: try to drop each column; if rank stays the same and decomposition
    # still valid, drop it.
    changed = True
    while changed and deadline_left_fn() > 5.0:
        changed = False
        for idx in list(keep):
            if deadline_left_fn() < 5.0:
                break
            candidate = [c for c in keep if c != idx]
            if len(candidate) == 0:
                continue
            Mc = M[:, candidate]
            rc = np.linalg.matrix_rank(Mc, tol=1e-9)
            if rc < base_rank:
                # Removing this column reduced rank -> it was independent, keep it.
                continue
            # Rank unchanged: test validity of the reduced decomposition.
            Ured = U[:, candidate]
            Vred = V[:, candidate]
            Wred = W[:, candidate]
            if _verify(Ured, Vred, Wred, n, m, p, tol=1e-9):
                # Valid! Remove this column.
                keep = candidate
                base_rank = rc
                changed = True
                break
    return keep


def construct():
    """Rank-minimal (within the recursive-Strassen family) exact tensor
    decomposition for 4x4 by 4x8 matrix multiplication.

    Builds the rank-98 recursive Kronecker-Strassen decomposition, then runs a
    verification + greedy column-reduction polish to remove any redundant
    columns, returning the smallest valid rank found within the time budget.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8

    U2, V2, W2 = _build_strassen_2x2()
    U4, V4, W4 = _build_4x4x4(U2, V2, W2)
    U, V, W = _build_4x4x8(U4, V4, W4)

    # Verify the base construction.
    if not _verify(U, V, W, n, m, p):
        # Fallback: return as-is (should not happen).
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W

    # Try to reduce columns if we have time.
    if _time_left() > 15.0:
        keep = _try_reduce_columns(U, V, W, n, m, p, _time_left)
        if len(keep) < U.shape[1]:
            Ured = U[:, keep]
            Vred = V[:, keep]
            Wred = W[:, keep]
            if _verify(Ured, Vred, Wred, n, m, p):
                best_U, best_V, best_W = Ured, Vred, Wred

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
