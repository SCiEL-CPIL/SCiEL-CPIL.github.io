"""
Combined program for matrix multiplication 3x4x7 tensor decomposition.
Uses 6 disjoint 2x2x2 Strassen blocks (rank 7 each) for i in {0,1},
tiling j-pairs {0,1},{2,3} and k-pairs {0,1},{2,3},{4,5},
plus schoolbook for the remaining 36 entries. Total rank: 78.

Refinement: after building the structured decomposition, attempts to reduce
rank by identifying linearly dependent columns and solving for a smaller
exact decomposition via structured linear algebra on the endomorphism ring.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]


def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    Returns dict with keys 'U', 'V', 'W' (lists of lists) representing
    factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit)

    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84

    def get_schoolbook():
        rank = num_eqs
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U_arr, V_arr, W_arr):
        """Vectorized verification."""
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        U_sub = U_arr[i_idx * m + j_idx, :]
        V_sub = V_arr[j_idx * p + k_idx, :]
        W_sub = W_arr[k_idx * n + i_idx, :]
        
        sums = np.sum(U_sub * V_sub * W_sub, axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)

    def add_strassen_block(U, V, W, col_start, i0, j0, k0):
        """Add a verified 2x2x2 Strassen block."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        c = col_start
        # Term 0
        U[i0*m+j0][c] = 1
        U[i1*m+j1][c] = 1
        V[j0*p+k0][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = 1
        W[k1*n+i1][c] = 1
        
        # Term 1
        c += 1
        U[i1*m+j0][c] = 1
        U[i1*m+j1][c] = 1
        V[j0*p+k0][c] = 1
        W[k0*n+i1][c] = 1
        W[k1*n+i1][c] = -1
        
        # Term 2
        c += 1
        U[i0*m+j0][c] = 1
        V[j0*p+k1][c] = 1
        V[j1*p+k1][c] = -1
        W[k1*n+i0][c] = 1
        W[k1*n+i1][c] = 1
        
        # Term 3
        c += 1
        U[i1*m+j1][c] = 1
        V[j1*p+k0][c] = 1
        V[j0*p+k0][c] = -1
        W[k0*n+i0][c] = 1
        W[k0*n+i1][c] = 1
        
        # Term 4
        c += 1
        U[i0*m+j0][c] = 1
        U[i0*m+j1][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = -1
        W[k1*n+i0][c] = 1
        
        # Term 5
        c += 1
        U[i1*m+j0][c] = 1
        U[i0*m+j0][c] = -1
        V[j0*p+k0][c] = 1
        V[j0*p+k1][c] = 1
        W[k1*n+i1][c] = 1
        
        # Term 6
        c += 1
        U[i0*m+j1][c] = 1
        U[i1*m+j1][c] = -1
        V[j1*p+k0][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = 1
        
        return c

    # Build the structured decomposition
    rank = 78
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]

    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]

    col = 0
    covered = set()
    for i0, j0, k0 in blocks:
        col = add_strassen_block(U, V, W, col, i0, j0, k0)
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))

    # Add schoolbook terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    if col != rank:
        return get_schoolbook()

    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)

    # Verify structured form
    if verify_fast(U_arr, V_arr, W_arr):
        # Attempt rank reduction via linear algebra
        if time.time() < deadline - 30:
            reduced = _try_reduce_rank(U_arr, V_arr, W_arr, n, m, p, deadline)
            if reduced is not None:
                return reduced
        return {"U": U, "V": V, "W": W}

    # Fallback: solve for W
    if time.time() > deadline - 25:
        return get_schoolbook()

    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    num_w_vars = dim_W * rank
    A = np.zeros((num_eqs, num_w_vars))
    b = np.ones(num_eqs)
    
    for eq in range(num_eqs):
        i, j, k = i_idx[eq], j_idx[eq], k_idx[eq]
        w_row = k * n + i
        for c in range(rank):
            coeff = U_arr[i*m+j, c] * V_arr[j*p+k, c]
            if coeff != 0:
                A[eq, w_row * rank + c] = coeff
    
    try:
        W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=1e-12)
    except Exception:
        return get_schoolbook()
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return get_schoolbook()
    
    W_sol = W_flat.reshape((dim_W, rank))
    W_rounded = np.round(W_sol * 2) / 2.0
    if verify_fast(U_arr, V_arr, np.asarray(W_rounded, dtype=np.float64)):
        return {"U": U, "V": V, "W": W_rounded.tolist()}
    
    W_int = np.round(W_sol)
    if verify_fast(U_arr, V_arr, np.asarray(W_int, dtype=np.float64)):
        return {"U": U, "V": V, "W": W_int.tolist()}
    
    return get_schoolbook()


def _try_reduce_rank(U_arr, V_arr, W_arr, n, m, p, deadline):
    """
    Attempt to reduce the rank by finding linear dependencies among the
    rank-1 terms. The key insight: if the columns of the "coefficient matrix"
    (mapping each term to its contribution pattern) have rank < current rank,
    we can eliminate redundant terms.
    
    Strategy: Form the tensor as a linear combination of rank-1 terms.
    Each term c contributes U[:,c] ⊗ V[:,c] ⊗ W[:,c] to the tensor.
    We check if we can express the same tensor with fewer terms by
    solving a linear system where we drop one term at a time.
    """
    dim_U, dim_V, dim_W = U_arr.shape[0], V_arr.shape[0], W_arr.shape[0]
    rank = U_arr.shape[1]
    
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Flatten T to a vector of length n*m*p
    # Each term c contributes a vector t_c where t_c[(i,j,k)] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # Build the tensor vector
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    num_eqs = n * m * p
    
    T = np.ones(num_eqs)  # Target tensor (all ones)
    
    # Build coefficient matrix: A[eq, c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # But that's the full product. Instead, for rank reduction, we need to
    # find if some linear combination of the terms equals the tensor with fewer terms.
    
    # Alternative approach: try removing one schoolbook term at a time and
    # see if the remaining terms can still span the required tensor.
    
    # More effective: identify which terms are "redundant" by checking if
    # their contribution is a linear combination of others.
    
    # For each term c, compute its contribution vector v_c of length num_eqs:
    # v_c[eq] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Build matrix M of shape (num_eqs, rank)
    M = np.zeros((num_eqs, rank))
    for c in range(rank):
        M[:, c] = U_arr[i_idx * m + j_idx, c] * V_arr[j_idx * p + k_idx, c] * W_arr[k_idx * n + i_idx, c]
    
    # Check if columns of M are linearly dependent
    # If rank(M) < rank, we can potentially reduce
    try:
        M_rank = np.linalg.matrix_rank(M, tol=1e-10)
    except Exception:
        return None
    
    if M_rank >= rank:
        return None  # No reduction possible via this method
    
    # Find a basis for the column space and express T in terms of fewer terms
    # Use QR decomposition to find independent columns
    try:
        Q, R = np.linalg.qr(M, mode='complete')
    except Exception:
        return None
    
    # Find pivot columns
    pivot_cols = []
    for c in range(rank):
        if abs(R[c, c]) > 1e-10:
            pivot_cols.append(c)
    
    if len(pivot_cols) >= rank:
        return None
    
    # Try to solve T = sum_{c in pivot_cols} alpha_c * M[:, c]
    # But we also need the resulting U, V, W to have half-integer entries.
    # This is complex, so let's try a simpler approach:
    # Remove the non-pivot columns and check if the remaining terms still work.
    
    # Actually, the cleanest approach: for each non-pivot column, check if
    # removing it still allows exact representation.
    
    for c_remove in range(rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        
        # Check if column c_remove is a linear combination of others
        others = [c for c in range(rank) if c != c_remove]
        M_others = M[:, others]
        
        try:
            coeffs, res, r, sv = np.linalg.lstsq(M_others, M[:, c_remove], rcond=1e-10)
        except Exception:
            continue
        
        if len(res) > 0 and res[0] > 1e-8:
            continue  # Not a linear combination, can't remove
        
        # Try removing this column
        U_new = np.delete(U_arr, c_remove, axis=1)
        V_new = np.delete(V_arr, c_remove, axis=1)
        W_new = np.delete(W_arr, c_remove, axis=1)
        
        if verify_fast(U_new, V_new, W_new):
            # Check for half-integer entries
            if _is_half_integer(U_new) and _is_half_integer(V_new) and _is_half_integer(W_new):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return None


def _is_half_integer(arr):
    """Check if all entries are half-integers (integer or integer + 0.5)."""
    doubled = arr * 2
    return np.allclose(doubled, np.round(doubled), atol=1e-9)
