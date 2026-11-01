import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s kill


def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _recon(Ua, Va, Wa):
    return np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)


def _validate(U, V, W, n, m, p, T=None):
    if T is None:
        T = _build_tensor(n, m, p)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T_recon = _recon(Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-9)


def _build_rank_32(n, m, p):
    """Rank-32: 4 Strassen 2x2x2 blocks (28 terms) + 1 schoolbook 2x2x1 block (4 terms)."""
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    strassen_terms = [
        # P0 = (A00+A11)*(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1), (1, -1), (2, 1)]),
        # P1 = (A10+A11)*B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # P2 = A00*(B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # P3 = A11*(B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, -1)]),
        # P4 = (A00+A01)*B11
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        # P5 = (A10-A00)*(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (2, 1)]),
        # P6 = (A01-A11)*(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]

    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx, col] += coeff
            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx, col] += coeff

    # Schoolbook for column 8
    schoolbook_terms = [
        ([(0, 1)], [(8, 1)], [(16, 1)]),
        ([(1, 1)], [(17, 1)], [(16, 1)]),
        ([(2, 1)], [(8, 1)], [(17, 1)]),
        ([(3, 1)], [(17, 1)], [(17, 1)]),
    ]
    col_start = 28
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff

    return U, V, W


def _try_remove_term(U, V, W, T, r, n, m, p):
    """Check if removing term r still gives a valid decomposition."""
    mask = [i for i in range(U.shape[1]) if i != r]
    T_recon = _recon(U[:, mask], V[:, mask], W[:, mask])
    if np.allclose(T, T_recon, atol=1e-9):
        return (U[:, mask].tolist(), V[:, mask].tolist(), W[:, mask].tolist())
    return None


def _search_rank_31_via_dependency(U32, V32, W32, T, n, m, p, budget):
    """
    Check if the rank-32 terms have a linear dependency.
    If sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0, then we can eliminate one term
    by re-expressing it. More directly: try removing each term.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > budget:
        return None

    # First: try simple single-term removal
    for r in range(32):
        if time.time() - _START_TIME > budget:
            return None
        result = _try_remove_term(U32, V32, W32, T, r, n, m, p)
        if result is not None:
            return result

    # Second: look for a linear dependency among the 32 rank-1 tensors.
    # Flatten each term to a vector, check if they are linearly dependent.
    # If dependent, there exists c != 0 with sum c_r * t_r = 0.
    # This means we can express one term as a combination of others,
    # but that doesn't directly give a rank-31 decomposition (the
    # decomposition requires the SAME coefficient structure).
    # However, if one term is a scalar multiple of another in a way that
    # allows merging, we could reduce rank.
    #
    # More practically: try pairwise merging. If term a and term b can be
    # combined into a single rank-1 term, we reduce rank by 1.
    # This requires u_a*v_a*w_a + u_b*v_b*w_b = u'*v'*w' (rank-1).
    # This is rare, so skip.
    #
    # Instead, try: perturb one term and solve for the rest.
    # This is the "absorb" approach.
    return None


def _try_block_11(n, m, p, budget):
    """
    Try to find a rank-11 decomposition for M_{2,2,3} (2x2 by 2x3).
    If found, use 3 copies for M_{2,2,9} giving rank 33.
    Then try to reduce 33 to 32 or 31.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > budget:
        return None

    # Build M_{2,2,3} tensor
    n2, m2, p2 = 2, 2, 3
    T23 = _build_tensor(n2, m2, p2)

    # Rank-10: Strassen (7) + schoolbook for 1 col (3 terms? No, 2x2x1 = 4 terms)
    # Actually 2x2x3: 2 Strassen blocks (14 terms) + 1 schoolbook (4 terms) = 18
    # Or 1 Strassen (7) + 2 schoolbook (8) = 15
    # Or 3 schoolbook = 12
    # Best known for 2x2x3 is 11 (from literature).
    # Let's try to construct rank-11 for 2x2x3.
    #
    # Known: M_{2,2,2} has rank 7 (Strassen).
    # M_{2,2,3} = M_{2,2,2} + M_{2,2,1} -> rank <= 7+4 = 11.
    # This is the trivial upper bound. Is it tight? Unknown, but let's build it.
    
    # Build rank-11 for 2x2x3: Strassen for cols 0-1 (7 terms) + schoolbook for col 2 (4 terms)
    rank23 = 11
    U23 = np.zeros((n2 * m2, rank23), dtype=float)
    V23 = np.zeros((m2 * p2, rank23), dtype=float)
    W23 = np.zeros((p2 * n2, rank23), dtype=float)

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1), (1, -1), (2, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (2, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]

    for t in range(7):
        col = t
        u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
        for u_idx, coeff in u_coeffs:
            U23[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V23[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W23[w_idx, col] += coeff

    # Schoolbook for col 2
    schoolbook_terms = [
        ([(0, 1)], [(2, 1)], [(4, 1)]),
        ([(1, 1)], [(5, 1)], [(4, 1)]),
        ([(2, 1)], [(2, 1)], [(5, 1)]),
        ([(3, 1)], [(5, 1)], [(5, 1)]),
    ]
    for t in range(4):
        col = 7 + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]
        for u_idx, coeff in u_coeffs:
            U23[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V23[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W23[w_idx, col] += coeff

    # Validate rank-11 for 2x2x3
    T_recon = _recon(U23, V23, W23)
    if not np.allclose(T23, T_recon, atol=1e-9):
        return None

    # Try to reduce rank-11 to rank-10 for 2x2x3
    for r in range(11):
        if time.time() - _START_TIME > budget:
            return None
        result = _try_remove_term(U23, V23, W23, T23, r, n2, m2, p2)
        if result is not None:
            # Found rank-10 for 2x2x3! Use 3 copies for 2x2x9 -> rank 30
            U10, V10, W10 = result
            # Build rank-30 for 2x2x9
            U9 = np.zeros((n * m, 30), dtype=float)
            V9 = np.zeros((m * p, 30), dtype=float)
            W9 = np.zeros((p * n, 30), dtype=float)
            for block in range(3):
                col_start = block * 10
                U9[:, col_start:col_start+10] = U10
                # V indices: for block b, V indices are b*3 + j for j in 0..2
                # But V10 has indices 0..5 (m2*p2 = 6). For 2x2x9, V has 18 rows.
                # V10 row j corresponds to (j_row, j_col) in 2x3.
                # For block b, V row j in 2x3 maps to V9 row b*3 + j
                for j in range(m2 * p2):
                    V9[block * p2 + j, col_start:col_start+10] = V10[j]
                # W indices: W10 has 6 rows (p2*n2). For block b, W row k maps to W9 row b*2 + k
                for k in range(p2 * n2):
                    W9[block * n2 + k, col_start:col_start+10] = W10[k]
            
            T = _build_tensor(n, m, p)
            T_recon = _recon(U9, V9, W9)
            if np.allclose(T, T_recon, atol=1e-9):
                return (U9.tolist(), V9.tolist(), W9.tolist())
            # Even if 3 copies don't work, try to reduce further
            # But first, just return rank-30 if valid
            return None

    # No rank-10 found. Use rank-11 for 2x2x3 -> rank-33 for 2x2x9
    U9 = np.zeros((n * m, 33), dtype=float)
    V9 = np.zeros((m * p, 33), dtype=float)
    W9 = np.zeros((p * n, 33), dtype=float)
    for block in range(3):
        col_start = block * 11
        U9[:, col_start:col_start+11] = U23
        for j in range(m2 * p2):
            V9[block * p2 + j, col_start:col_start+11] = V23[j]
        for k in range(p2 * n2):
            W9[block * n2 + k, col_start:col_start+11] = W23[k]
    
    T = _build_tensor(n, m, p)
    T_recon = _recon(U9, V9, W9)
    if not np.allclose(T, T_recon, atol=1e-9):
        return None

    # Try to reduce rank-33 to rank-32
    for r in range(33):
        if time.time() - _START_TIME > budget:
            return None
        result = _try_remove_term(U9, V9, W9, T, r, n, m, p)
        if result is not None:
            return result

    return None


def _try_local_perturbation(U32, V32, W32, T, n, m, p, budget):
    """
    Try to find a rank-31 decomposition by perturbing the rank-32 solution.
    Idea: fix 31 terms, solve for the 32nd, check if it can be made zero.
    Or: perturb all terms slightly and use gradient descent to drive one term to zero.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > budget:
        return None

    # Approach: for each target term r, try to make it zero by adjusting others.
    # This is a nonlinear system. Use a simple iterative approach.
    #
    # More practical: try small random perturbations of one term and check
    # if the remaining 31 terms can be adjusted to compensate.
    # This is expensive, so limit the search.
    
    # Try a different approach: look for a linear combination of terms that
    # gives zero contribution to the tensor.
    # If sum_{r in S} c_r * (u_r ⊗ v_r ⊗ w_r) = 0, then we can remove one term
    # from S and adjust the others.
    #
    # Check all pairs: can two terms be combined into one?
    # u_a*v_a*w_a + u_b*v_b*w_b = u'*v'*w'
    # This is a rank-1 condition on the sum.
    for a in range(32):
        if time.time() - _START_TIME > budget:
            return None
        for b in range(a + 1, 32):
            # Check if u_a⊗v_a⊗w_a + u_b⊗v_b⊗w_b is rank-1
            # Flatten to a 4*18*18 vector
            T_ab = np.zeros((n * m, m * p, p * n), dtype=float)
            T_ab += np.einsum('i,j,k->ijk', U32[:, a], V32[:, a], W32[:, a])
            T_ab += np.einsum('i,j,k->ijk', U32[:, b], V32[:, b], W32[:, b])
            # Check if T_ab is rank-1: all 2x2 minors are zero
            # For a 4x18x18 tensor, rank-1 means T_ab = u'⊗v'⊗w'
            # Check: for any two i indices, the 18x18 slices are proportional
            flat = T_ab.reshape(n * m, -1)  # 4 x 324
            # Rank of flat should be 1
            sv = np.linalg.svd(flat, compute_uv=False)
            if len(sv) > 1 and sv[1] < 1e-9:
                # T_ab is rank-1 in the first mode, but need full rank-1
                # Check all modes
                flat2 = T_ab.reshape(m * p, -1)  # 18 x 72
                sv2 = np.linalg.svd(flat2, compute_uv=False)
                if len(sv2) > 1 and sv2[1] < 1e-9:
                    flat3 = T_ab.reshape(p * n, -1)  # 18 x 72
                    sv3 = np.linalg.svd(flat3, compute_uv=False)
                    if len(sv3) > 1 and sv3[1] < 1e-9:
                        # Found a rank-1 combination! Remove one term.
                        # The combined term is T_ab, which equals u'⊗v'⊗w'
                        # We can replace terms a and b with a single term.
                        # But we need to extract u', v', w' from T_ab.
                        # For now, just try removing term a and b and adding the combined term.
                        # Actually, this would keep rank the same (remove 2, add 1 = net -1).
                        # Let's just return a rank-31 by removing one of them.
                        # This is approximate; need to be careful.
                        pass

    return None


def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME

    T = _build_tensor(n, m, p)

    # Build rank-32 baseline
    U32, V32, W32 = _build_rank_32(n, m, p)
    if not _validate(U32.tolist(), V32.tolist(), W32.tolist(), n, m, p, T):
        # Fallback to schoolbook rank-36
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank-31 via dependency search
    if time.time() - _START_TIME < _TIMEOUT - 20:
        result = _search_rank_31_via_dependency(U32, V32, W32, T, n, m, p, _TIMEOUT - 20)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p, T):
                return {"U": U, "V": V, "W": W}

    # Try block-11 approach
    if time.time() - _START_TIME < _TIMEOUT - 15:
        result = _try_block_11(n, m, p, _TIMEOUT - 15)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p, T):
                return {"U": U, "V": V, "W": W}

    # Try local perturbation
    if time.time() - _START_TIME < _TIMEOUT - 10:
        result = _try_local_perturbation(U32, V32, W32, T, n, m, p, _TIMEOUT - 10)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p, T):
                return {"U": U, "V": V, "W": W}

    # Return rank-32
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
