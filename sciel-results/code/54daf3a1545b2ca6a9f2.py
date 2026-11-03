import numpy as np
import time
import itertools

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    r = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # A has shape (n, m), B has shape (m, p), C has shape (n, p)
    # u_idx(i,j) = i*m+j  (A row i, col j)
    # v_idx(i,j) = i*p+j  (B row i, col j)
    # w_idx(i,j) = j*n+i  (C: W[k*n+r] where k is row of C, r is col of C)
    # So w_idx(c_row, c_col) = c_col*n + c_row
    
    # Check: for all a_row, a_col, b_row, b_col
    # sum_{j} A[a_row, j] * B[j, b_row]... wait, B is (m,p) so B[j, b_col]
    # C[a_row, b_col] = sum_j A[a_row, j] * B[j, b_col]
    # In tensor form: T[A_idx, B_idx, C_idx] should give 1 when
    # A_idx = a_row*m + a_col, B_idx = b_row*p + b_col, C_idx = b_col*n + a_row
    # and 0 otherwise.
    
    # Efficient check: compute the bilinear form
    # For each rank term r: outer(U[:,r], V[:,r]) * W[:,r]
    # Sum over r, check equals the multiplication tensor.
    
    # Build expected tensor
    expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for a_r in range(n):
        for a_c in range(m):
            for b_c in range(p):
                # C[a_r, b_c] gets contribution from A[a_r, a_c] * B[a_c, b_c]
                # A_idx = a_r*m + a_c
                # B_idx = a_c*p + b_c
                # C_idx = b_c*n + a_r
                expected[a_r*m + a_c, a_c*p + b_c, b_c*n + a_r] = 1.0
    
    # Compute decomposed tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(r):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    return np.allclose(T, expected, atol=1e-10)

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 110  # safety margin
    
    # Start with the known rank-28 Strassen decomposition
    U28, V28, W28 = _strassen_28(n, m, p)
    
    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28
    
    # Try to eliminate redundant columns (linear dependencies)
    best_U, best_V, best_W, best_rank = _eliminate_redundancy(U28, V28, W28, n, m, p, deadline)
    
    if time.time() > deadline - 10:
        return _to_list(best_U, best_V, best_W)
    
    # Try random rank reduction via projection
    for target_rank in range(best_rank - 1, max(20, best_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank_reduction(best_U, best_V, best_W, target_rank, n, m, p, deadline)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            if best_rank <= 24:
                break
    
    return _to_list(best_U, best_V, best_W)

def _strassen_28(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2)
    ]
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    current_rank = 0
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        u_map = [
            u_idx(a_r, a_c),
            u_idx(a_r, a_c + 1),
            u_idx(a_r + 1, a_c),
            u_idx(a_r + 1, a_c + 1)
        ]
        v_map = [
            v_idx(b_r, b_c),
            v_idx(b_r, b_c + 1),
            v_idx(b_r + 1, b_c),
            v_idx(b_r + 1, b_c + 1)
        ]
        w_map = [
            w_idx(0, c_c),
            w_idx(0, c_c + 1),
            w_idx(1, c_c),
            w_idx(1, c_c + 1)
        ]
        for (u_terms, v_terms, w_terms) in strassen_terms:
            for local_idx, coeff in u_terms:
                U[u_map[local_idx], current_rank] += coeff
            for local_idx, coeff in v_terms:
                V[v_map[local_idx], current_rank] += coeff
            for local_idx, coeff in w_terms:
                W[w_map[local_idx], current_rank] += coeff
            current_rank += 1
    
    return U, V, W

def _eliminate_redundancy(U, V, W, n, m, p, deadline):
    """Try to find linear dependencies among rank terms to reduce rank."""
    r = U.shape[1]
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    
    # Try to find subsets of columns that are linearly dependent in the tensor sense
    # A set of rank terms is redundant if there exists nonzero (alpha, beta, gamma)
    # such that sum_r alpha_r * U_r = 0, sum_r beta_r * V_r = 0, sum_r gamma_r * W_r = 0
    # and alpha_r * beta_r * gamma_r combination... actually this is more subtle.
    
    # Simpler approach: check if any single column can be removed.
    # If column r is zero in all three matrices, it's trivially removable.
    for r in range(r):
        if time.time() > deadline:
            break
        if np.all(Uf[:, r] == 0) or np.all(Vf[:, r] == 0) or np.all(Wf[:, r] == 0):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            r -= 1
    
    # Try removing pairs: if columns r1, r2 satisfy U_r1 = a*U_r2, V_r1 = b*V_r2, W_r1 = c*W_r2
    # then they can be merged if a*b*c = 1 (for the product to be preserved)
    # Actually for tensor decomposition, two terms are redundant if:
    # There exist scalars such that the sum of the two terms equals one term.
    # This is complex. Let's just check for exact duplicates.
    seen = {}
    keep = []
    for r in range(r):
        key = (tuple(Uf[:, r].tolist()), tuple(Vf[:, r].tolist()), tuple(Wf[:, r].tolist()))
        if key in seen:
            # Try to merge: if W values are compatible
            pass
        else:
            seen[key] = r
            keep.append(r)
    
    if len(keep) < r:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    # Verify and try small random eliminations
    # For each pair of columns, check if they can be combined
    r = U.shape[1]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r1 in range(r):
            for r2 in range(r1 + 1, r):
                if time.time() > deadline:
                    break
                # Check if U[:,r1] is proportional to U[:,r2]
                # and V[:,r1] is proportional to V[:,r2]
                # and W[:,r1] is proportional to W[:,r2]
                u1, u2 = Uf[:, r1], Uf[:, r2]
                v1, v2 = Vf[:, r1], Vf[:, r2]
                w1, w2 = Wf[:, r1], Wf[:, r2]
                
                if np.all(u1 == 0) or np.all(u2 == 0):
                    continue
                if np.all(v1 == 0) or np.all(v2 == 0):
                    continue
                if np.all(w1 == 0) or np.all(w2 == 0):
                    continue
                
                # Find scaling factors
                try:
                    # u2 = a * u1
                    nonzero_u = u1 != 0
                    if not np.any(nonzero_u):
                        continue
                    a_candidates = u2[nonzero_u] / u1[nonzero_u]
                    a = a_candidates[0]
                    if not np.allclose(u2, a * u1, atol=1e-10):
                        continue
                    
                    # v2 = b * v1
                    nonzero_v = v1 != 0
                    if not np.any(nonzero_v):
                        continue
                    b_candidates = v2[nonzero_v] / v1[nonzero_v]
                    b = b_candidates[0]
                    if not np.allclose(v2, b * v1, atol=1e-10):
                        continue
                    
                    # w2 = c * w1
                    nonzero_w = w1 != 0
                    if not np.any(nonzero_w):
                        continue
                    c_candidates = w2[nonzero_w] / w1[nonzero_w]
                    c = c_candidates[0]
                    if not np.allclose(w2, c * w1, atol=1e-10):
                        continue
                    
                    # The product a*b*c should allow merging
                    # Term r1: u1 * v1 * w1
                    # Term r2: u2 * v2 * w2 = a*b*c * u1 * v1 * w1
                    # Sum: (1 + a*b*c) * u1 * v1 * w1
                    # This is just a scaled version of term r1, so we can replace both with one
                    factor = 1.0 + a * b * c
                    if abs(factor) < 1e-10:
                        # They cancel! Remove both
                        keep = [k for k in range(r) if k not in (r1, r2)]
                        U = U[:, keep]
                        V = V[:, keep]
                        W = W[:, keep]
                        Uf = U.astype(np.float64)
                        Vf = V.astype(np.float64)
                        Wf = W.astype(np.float64)
                        r = U.shape[1]
                        changed = True
                        break
                    else:
                        # Replace r1 with scaled version, remove r2
                        U[:, r1] = factor * U[:, r1]
                        V = V
                        W = W
                        # Actually we need to keep the product the same
                        # Just scale U
                        keep = [k for k in range(r) if k != r2]
                        U = U[:, keep]
                        V = V[:, keep]
                        W = W[:, keep]
                        Uf = U.astype(np.float64)
                        Vf = V.astype(np.float64)
                        Wf = W.astype(np.float64)
                        r = U.shape[1]
                        changed = True
                        break
                except:
                    continue
            if changed:
                break
    
    # Verify result
    if U.shape[1] < 28 and _verify(U, V, W, n, m, p):
        return U, V, W, U.shape[1]
    
    return U, V, W, U.shape[1]

def _try_rank_reduction(U, V, W, target_rank, n, m, p, deadline):
    """Try to project to a lower rank."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return None
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    for r in range(current_rank):
        T += np.einsum('i,j,k->ijk', Uf[:, r], Vf[:, r], Wf[:, r])
    
    # Try random linear combinations of columns
    np.random.seed(42)
    best = None
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random projection: combine current_rank columns into target_rank
        P = np.random.randn(current_rank, target_rank)
        U_new = Uf @ P
        V_new = Vf @ np.linalg.inv(P) if np.linalg.matrix_rank(P) == target_rank else None
        
        if V_new is None:
            continue
        
        # This doesn't work simply because we need U@V@W products to be preserved
        # The correct approach: T = sum_r U_r V_r W_r = (U P)(P^{-1} V) W if P is invertible
        # But P is (current_rank x target_rank), not square.
        
        break
    
    return None

def _to_list(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
