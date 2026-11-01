import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _build_rank28():
    """Build the rank-28 Strassen-block decomposition. Returns (U, V, W) as int arrays."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def strassen_terms():
        terms = []
        # M1
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        # M2
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        # M3
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        # M4
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        # M5
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        # M6
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        # M7
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()

    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * 4 + (a_offset * 2 + c)
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * 4 + (b_col_offset * 2 + c)
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_offset * 2 + c) * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            for idx, coeff in u_global.items():
                U[idx, col] += coeff
            for idx, coeff in v_global.items():
                V[idx, col] += coeff
            for idx, coeff in w_global.items():
                W[idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    return U, V, W

def _verify(U, V, W):
    """Check exact decomposition on random integer matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-6):
            return False
    return True

def _search_rank27(U28, V28, W28):
    """
    Search for a rank-27 decomposition by finding a linear combination
    of the 28 rank-1 terms that cancels one term.
    
    We look for vectors a, b, c (one for each of U, V, W terms) such that
    the 29th term (sum) equals one of the existing terms, allowing cancellation.
    
    Actually, the approach: we have 28 rank-1 terms. We want to find a
    non-trivial linear relation. If sum_i alpha_i * term_i = 0 (as a tensor),
    then we can drop any term with alpha_i != 0 by redistributing.
    
    More practically: search for a vector v in R^28 such that
    sum_i v_i * (u_i, v_i, w_i) = 0 as a 3-tensor.
    This is a system of n*m*p = 32 equations in 28 unknowns (overdetermined),
    so generally no solution. But we can search for approximate solutions
    or look at the structure.
    
    Alternative: try to find two terms that are linearly dependent in some
    structured way. Or search over small integer combinations.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # Build the 3-tensor for each rank-1 term
    # Tensor T_i has shape (n*m, m*p, p*n)
    # T_i[a, b, c] = U[a,i] * V[b,i] * W[c,i]
    
    # We want to find v in R^28 such that sum_i v_i * T_i = 0
    # This is a homogeneous linear system. The number of constraints is n*m*p = 32.
    # Since 32 > 28, the system is overdetermined and typically has only the
    # trivial solution. But let's check the rank of the constraint matrix.
    
    # Flatten each T_i to a vector of length 32
    T_flat = np.zeros((rank, n * m * p), dtype=float)
    for i in range(rank):
        u_vec = U28[:, i].astype(float)
        v_vec = V28[:, i].astype(float)
        w_vec = W28[:, i].astype(float)
        tensor = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        T_flat[i, :] = tensor.reshape(-1)
    
    # Find null space of T_flat (shape 28 x 32)
    # We want v such that T_flat @ v = 0
    # Use SVD
    try:
        U_svd, s, Vt = np.linalg.svd(T_flat.T, full_matrices=True)
        # Null space of T_flat is spanned by rows of Vt corresponding to zero singular values
        # T_flat is 28x32, so T_flat.T is 32x28
        # SVD: T_flat.T = U_svd @ diag(s) @ Vt
        # Null space of T_flat = {v : T_flat @ v = 0}
        # = {v : T_flat.T @ v = 0}... wait, T_flat is 28x32
        # We want v in R^28 such that T_flat @ v = 0 (32 equations, 28 unknowns)
        # SVD of T_flat: T_flat = U_svd @ diag(s) @ Vt, where Vt is 28x28
        # Null space is spanned by rows of Vt where s is zero
        U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=True)
        # s has length min(28, 32) = 28
        # Find indices where s is close to zero
        tol = 1e-10
        null_indices = np.where(s < tol)[0]
        if len(null_indices) > 0:
            # Null space is spanned by Vt[null_indices, :]
            null_space = Vt[null_indices, :]
            # Check if there's a non-trivial solution
            # For each basis vector in null space, try to use it
            for idx in null_indices:
                v = Vt[idx, :]
                # Check if v has at least 2 non-zero entries (for a meaningful relation)
                nz = np.where(np.abs(v) > 1e-8)[0]
                if len(nz) >= 2:
                    # Try to construct a rank-27 decomposition using this relation
                    # If sum_i v_i * term_i = 0, we can drop term j by
                    # redistributing: term_j' = term_j - (v_j / v_k) * term_k for some k
                    # But this changes the decomposition. Actually, if the relation holds,
                    # we can simply drop any term with v_i != 0 and adjust the others.
                    # More precisely: if v_j != 0, we can express term_j as
                    # -(1/v_j) * sum_{i != j} v_i * term_i
                    # So we can drop term j and the remaining 27 terms still decompose the tensor.
                    # Let's verify this.
                    j = nz[0]
                    if abs(v[j]) < 1e-10:
                        continue
                    # Build new decomposition without term j
                    U27 = np.delete(U28, j, axis=1)
                    V27 = np.delete(V28, j, axis=1)
                    W27 = np.delete(W28, j, axis=1)
                    # Verify
                    if _verify(U27, V27, W27):
                        return U27, V27, W27
    except np.linalg.LinAlgError:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-28 decomposition
    U28, V28, W28 = _build_rank28()
    
    # Verify rank-28 works
    if not _verify(U28, V28, W28):
        # Fallback: return rank-28 anyway (shouldn't happen)
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try to find a rank-27 decomposition
    if time.time() < _DEADLINE - 10.0:
        result = _search_rank27(U28, V28, W28)
        if result is not None:
            U27, V27, W27 = result
            return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Return rank-28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
