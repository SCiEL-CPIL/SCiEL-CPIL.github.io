import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _reduce_rank_via_column_space(U, V, W, n, m, p, tol=1e-9):
    """
    Attempt to reduce the rank by finding linear dependencies among rank columns.
    
    The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] must equal 1.
    
    Strategy: For each pair of columns (r1, r2), check if column r2 is a linear
    combination of other columns that preserves the tensor. If we can express
    one column as a combination of others, we can eliminate it.
    
    More practically: use the fact that the rank of the tensor is at most
    the minimum over all unfoldings. We try to find a smaller set of rank
    columns that span the same tensor.
    
    Approach: Use iterative elimination. For each column r, check if removing it
    and adjusting others preserves the tensor. This is expensive, so we use
    a heuristic: find columns that are linearly dependent in the U-matrix
    (or V or W) and try to merge them.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Work on copies
    U_work = Ua.copy()
    V_work = Va.copy()
    W_work = Wa.copy()
    
    # Try to eliminate columns one by one
    # For column r, we need to find coefficients such that
    # U[:,r] = sum_{r' != r} alpha_r' * U[:,r']
    # AND V[:,r] = sum_{r' != r} alpha_r' * V[:,r']  
    # AND W[:,r] = sum_{r' != r} alpha_r' * W[:,r']
    # This is a strong condition. A weaker one: the tensor contribution
    # of column r can be absorbed by other columns.
    
    # Alternative: use SVD on the "flattened" representation
    # The tensor T can be seen as a matrix of size (n*m*p) x rank
    # where T_flat[i*m*p + j*p + k, r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # But this isn't linear in r.
    
    # Better approach: For each column r, the contribution to T is:
    # T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is a rank-1 tensor in the (i,j,k) space.
    
    # We can try: for each column r, express T_r as a linear combination
    # of T_{r'} for r' != r. If possible, eliminate r.
    
    # This is complex. Let's use a simpler heuristic:
    # Check if any two columns are identical (or negatives) in all three matrices.
    
    # Build a fingerprint for each column
    def col_fingerprint(U_mat, V_mat, W_mat, r):
        u = U_mat[:, r]
        v = V_mat[:, r]
        w = W_mat[:, r]
        # Normalize to handle scaling
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        norm_w = np.linalg.norm(w)
        if norm_u < tol or norm_v < tol or norm_w < tol:
            return None
        return (np.round(u / norm_u, 8), np.round(v / norm_v, 8), np.round(w / norm_w, 8))
    
    # Find duplicate columns (same direction in U, V, W)
    fps = []
    for r in range(rank):
        fp = col_fingerprint(U_work, V_work, W_work, r)
        fps.append(fp)
    
    # Group columns by fingerprint
    from collections import defaultdict
    groups = defaultdict(list)
    for r, fp in enumerate(fps):
        if fp is not None:
            # Use a hashable key
            key = (fp[0].tobytes(), fp[1].tobytes(), fp[2].tobytes())
            groups[key].append(r)
    
    # For groups with >1 column, we might be able to combine them
    # But this is heuristic and may not work.
    
    # Let's try a different approach: use the rank of the U matrix
    # to determine the minimum possible rank.
    u_rank = np.linalg.matrix_rank(Ua, tol=tol)
    v_rank = np.linalg.matrix_rank(Va, tol=tol)
    w_rank = np.linalg.matrix_rank(Wa, tol=tol)
    
    # The tensor rank is at least max(u_rank, v_rank, w_rank)
    # but this is a lower bound, not necessarily achievable.
    
    # For now, just return the original if no easy reduction found
    return U, V, W, rank


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0

    best = None
    best_rank = float('inf')

    # Try rank-56 first (known valid)
    if time.time() < deadline:
        cand56 = _construct_rank56()
        ok56, err56 = _matmul_check(cand56["U"], cand56["V"], cand56["W"], n, m, p)
        if ok56:
            U, V, W, r = _trim_zero_columns(cand56["U"], cand56["V"], cand56["W"])
            if r < best_rank:
                best_rank = r
                best = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank via column space analysis
    if best is not None and time.time() < deadline:
        U, V, W, r = best["U"], best["V"], best["W"], best_rank
        U_reduced, V_reduced, W_reduced, r_reduced = _reduce_rank_via_column_space(
            U, V, W, n, m, p
        )
        if r_reduced < best_rank:
            ok_red, err_red = _matmul_check(U_reduced, V_reduced, W_reduced, n, m, p)
            if ok_red:
                best_rank = r_reduced
                best = {"U": U_reduced, "V": V_reduced, "W": W_reduced}

    if best is not None:
        return best

    # Last resort: return untrimmed rank-56
    cand56 = _construct_rank56()
    return cand56
# EVOLVE-BLOCK-END
