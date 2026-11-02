"""Matrix multiplication 4x4x4 tensor decomposition: rank-48 via linear dependence search."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Inner 2x2 Strassen pattern (rank 7).
# ---------------------------------------------------------------------------
_INNER_STRASSEN_2X2 = [
    {'u': [(0, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1), (1, 1, 1)]},
    {'u': [(1, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1)], 'w': [(1, 0, 1), (1, 1, -1)]},
    {'u': [(0, 0, 1)], 'v': [(0, 1, 1), (1, 1, -1)], 'w': [(0, 1, 1), (1, 1, 1)]},
    {'u': [(1, 1, 1)], 'v': [(1, 0, 1), (0, 0, -1)], 'w': [(0, 0, 1), (1, 0, 1)]},
    {'u': [(0, 0, 1), (0, 1, 1)], 'v': [(1, 1, 1)], 'w': [(0, 1, 1), (0, 0, -1)]},
    {'u': [(1, 0, 1), (0, 0, -1)], 'v': [(0, 0, 1), (0, 1, 1)], 'w': [(1, 1, 1)]},
    {'u': [(0, 1, 1), (1, 1, -1)], 'v': [(1, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1)]},
]

# ---------------------------------------------------------------------------
# Outer 4x4 Strassen pattern (rank 7) over 2x2 blocks.
# ---------------------------------------------------------------------------
_OUTER_STRASSEN_4X4 = [
    {'a': [(0, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1), (1, 1, 1)]},
    {'a': [(1, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1)], 'c': [(1, 0, 1), (1, 1, -1)]},
    {'a': [(0, 0, 1)], 'b': [(0, 1, 1), (1, 1, -1)], 'c': [(0, 1, 1), (1, 1, 1)]},
    {'a': [(1, 1, 1)], 'b': [(1, 0, 1), (0, 0, -1)], 'c': [(0, 0, 1), (1, 0, 1)]},
    {'a': [(0, 0, 1), (0, 1, 1)], 'b': [(1, 1, 1)], 'c': [(0, 1, 1), (0, 0, -1)]},
    {'a': [(1, 0, 1), (0, 0, -1)], 'b': [(0, 0, 1), (0, 1, 1)], 'c': [(1, 1, 1)]},
    {'a': [(0, 1, 1), (1, 1, -1)], 'b': [(1, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1)]},
]


def _strassen_4x4(n, m, p):
    """Build rank-49 U, V, W factor matrices via recursive Strassen."""
    assert (n, m, p) == (4, 4, 4)
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for s, outer in enumerate(_OUTER_STRASSEN_4X4):
        for t, inner in enumerate(_INNER_STRASSEN_2X2):
            r = s * 7 + t
            for (P, Q, sa) in outer['a']:
                for (ir, jc, sx) in inner['u']:
                    coeff = sa * sx
                    if coeff:
                        U[(2*P+ir)*m + (2*Q+jc), r] += coeff
            for (P, Q, sb) in outer['b']:
                for (ir, jc, sy) in inner['v']:
                    coeff = sb * sy
                    if coeff:
                        V[(2*P+ir)*p + (2*Q+jc), r] += coeff
            for (P, Q, sc) in outer['c']:
                for (ir, jc, sz) in inner['w']:
                    coeff = sc * sz
                    if coeff:
                        W[(2*Q+jc)*n + (2*P+ir), r] += coeff

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify: T[i,j,k] = sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1."""
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ikr,kjr,jir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook_decomposition(n, m, p):
    """Fallback rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    return U, V, W


def _find_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Search for a linear dependence among the 49 rank-1 terms that allows
    reducing the rank by 1. The tensor is T = sum_r u_r ⊗ v_r ⊗ w_r.
    If sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 with some c_r != 0, and the
    dependence allows re-expressing one term using others, we can reduce rank.
    
    More practically: we look for a vector c such that the 48x48 submatrix
    of the tensor's flattening has a specific structure. 
    
    Actually, the most direct approach: check if any single rank-1 term is
    linearly dependent on the others in the tensor space. The tensor space
    has dimension 64. We have 49 terms. If the 49 vectors (flattened rank-1
    tensors) span a space of dimension < 49, we can reduce.
    
    But even if they span 49-dim, there might be a "rank reduction" via
    a more sophisticated algebraic identity. Let's first check the simple
    linear dependence.
    """
    rank = U.shape[1]
    
    # Flatten each rank-1 term into a 64-dim vector
    # T_r[i,j,k] = U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Flatten as vector of length n*m*p = 64, indexed by (i,j,k) -> i*m*p + j*m + k
    # Wait, let's use a consistent flattening: index = (i*m + k)*p + j or similar.
    # Actually for checking linear dependence, any consistent flattening works.
    # Let's use: flat_index = i * (m*p) + k * p + j  (row-major over (i,k,j))
    
    dim = n * m * p  # 64
    T_flat = np.zeros((dim, rank), dtype=float)
    
    for r in range(rank):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * (m * p) + k * p + j
                    T_flat[idx, r] = U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # Check rank of T_flat (49 columns, 64 rows)
    # If rank < 49, there's a linear dependence.
    try:
        # SVD to find rank
        U_svd, s_vals, Vt = np.linalg.svd(T_flat, full_matrices=False)
        # Count significant singular values
        tol = 1e-10 * s_vals[0]
        num_rank = np.sum(s_vals > tol)
        
        if num_rank < rank:
            # There's a linear dependence! Find the null space vector.
            # The null space of T_flat (as a 64x49 matrix) gives coefficients c
            # such that T_flat @ c = 0, i.e., sum_r c_r * T_r = 0.
            # The null space has dimension rank - num_rank >= 1.
            # Vt has shape (min(64,49), 49) = (49, 49). Rows beyond num_rank are null space.
            null_vec = Vt[num_rank, :]  # first null space vector
            # Find the index with largest |c_r|
            idx_max = np.argmax(np.abs(null_vec))
            c = null_vec / null_vec[idx_max]
            
            # Now we have: sum_r c_r * T_r = 0
            # This means T_{idx_max} = -sum_{r != idx_max} (c_r / c_{idx_max}) * T_r
            # But this doesn't directly give a rank reduction in the tensor decomposition
            # sense, because the tensor decomposition requires the sum to equal the
            # multiplication tensor, not zero.
            
            # However, if the original sum equals the multiplication tensor M,
            # and sum c_r T_r = 0, then we can write:
            # M = sum_r T_r = sum_{r != idx_max} T_r + T_{idx_max}
            # And T_{idx_max} = -sum_{r != idx_max} (c_r/c_{idx_max}) T_r
            # So M = sum_{r != idx_max} [1 - c_r/c_{idx_max}] T_r
            # This gives a rank-(49-1) = 48 decomposition!
            
            # But wait - this only works if the dependence is "exact" in the sense
            # that the recombined coefficients still give a valid tensor decomposition.
            # Since tensor decomposition is just a sum of rank-1 tensors, and we're
            # just recombining the coefficients, it should work.
            
            # Let's verify: build the new 48-term decomposition.
            new_rank = rank - 1
            U_new = np.zeros((n * m, new_rank), dtype=float)
            V_new = np.zeros((m * p, new_rank), dtype=float)
            W_new = np.zeros((p * n, new_rank), dtype=float)
            
            col = 0
            for r in range(rank):
                if r == idx_max:
                    continue
                new_coeff = 1.0 - c[r] / c[idx_max]
                if abs(new_coeff) < 1e-15:
                    continue  # skip zero terms
                U_new[:, col] = U[:, r] * new_coeff
                V_new[:, col] = V[:, r] * new_coeff
                W_new[:, col] = W[:, r] * new_coeff
                col += 1
            
            actual_rank = col
            if actual_rank < new_rank:
                U_new = U_new[:, :actual_rank]
                V_new = V_new[:, :actual_rank]
                W_new = W_new[:, :actual_rank]
            
            # Verify
            if _verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-8):
                return U_new, V_new, W_new
            
    except Exception:
        pass
    
    return None


def construct():
    """
    Attempt rank-48 decomposition for 4x4 matrix multiplication.
    Falls back to rank-49 Strassen if no rank reduction is found.
    """
    t_start = time.time()
    deadline = t_start + 110.0

    n = m = p = 4

    # First: build rank-49 Strassen
    U, V, W = _strassen_4x4(n, m, p)
    
    # Try to find a rank-48 decomposition via linear dependence
    if time.time() < deadline:
        result = _find_rank_reduction(U, V, W, n, m, p, deadline)
        if result is not None:
            U48, V48, W48 = result
            if _verify_decomposition(U48, V48, W48, n, m, p, atol=1e-8):
                return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}
    
    # Fallback: rank-49 Strassen
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Last resort: schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
