"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_factors():
    factors = [
        {'U': {(0,0):1,(1,1):1}, 'V': {(0,0):1,(1,1):1}, 'W': {(0,0):1,(1,1):1}},
        {'U': {(1,0):1,(1,1):1}, 'V': {(0,0):1}, 'W': {(1,0):1}},
        {'U': {(0,0):1}, 'V': {(1,0):1,(0,0):-1}, 'W': {(0,1):1,(1,1):1}},
        {'U': {(1,1):1}, 'V': {(1,1):1,(0,1):-1}, 'W': {(0,0):1,(1,0):1}},
        {'U': {(0,0):1,(0,1):1}, 'V': {(1,1):1}, 'W': {(0,1):1,(1,1):-1}},
        {'U': {(1,0):1,(0,0):-1}, 'V': {(0,0):1,(0,1):1}, 'W': {(1,1):1}},
        {'U': {(0,1):1,(1,1):-1}, 'V': {(1,0):1,(1,1):1}, 'W': {(0,0):1,(1,1):1}},
    ]
    return factors

def _build_strassen_4x4_4x4():
    level1_factors = _strassen_2x2_factors()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    col = 0
    for l1 in range(7):
        for l2 in range(7):
            f1, f2 = level1_factors[l1], level1_factors[l2]
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in f1['U']: continue
                    c1 = f1['U'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in f2['U']: continue
                            c2 = f2['U'][(si, sj)]
                            U[bi*2+si, col] += c1 * c2
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in f1['V']: continue
                    c1 = f1['V'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in f2['V']: continue
                            c2 = f2['V'][(si, sj)]
                            V[bi*2+si, col] += c1 * c2
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in f1['W']: continue
                    c1 = f1['W'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in f2['W']: continue
                            c2 = f2['W'][(si, sj)]
                            W[bi*2+si, col] += c1 * c2
            col += 1
    return U, V, W

def _build_strassen_tiled(n, m, p):
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    rank_s = 49
    rank_sb = n * m * (p - 4)
    rank_total = rank_s + rank_sb
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    U[:, :rank_s] = U_s
    V[:16, :rank_s] = V_s
    W[:16, :rank_s] = W_s
    col = rank_s
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, trials=5, seed=12345):
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                Wrow = W[k*n+i]
                for j in range(m):
                    s += np.dot(U[i*m+j] * V[j*p+k], Wrow)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _verify_fast(U, V, W, n, m, p, seed=999):
    """Faster verification using tensor contraction."""
    rng = np.random.RandomState(seed)
    for _ in range(3):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        # result[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                Wr = W[k*n+i]
                for j in range(m):
                    acc += np.dot(U[i*m+j] * V[j*p+k], Wr)
                result[i, k] = acc
        if not np.allclose(result, expected, atol=1e-8):
            return False
    return True

def _try_reduce(U, V, W, n, m, p, max_time):
    """
    Try to reduce rank by finding linear dependencies among columns.
    Strategy: repeatedly try to express one column as a linear combination
    of others, or find pairs of columns that can be merged.
    """
    start = time.time()
    rank = U.shape[1]
    # Work with copies
    Uc = U.copy()
    Vc = V.copy()
    Wc = W.copy()
    
    # Try to find redundant columns using random projections
    # If column r can be written as linear combo of others, we can eliminate it.
    # A sufficient condition: for all (i,j,k), the "contribution" of column r
    # equals the sum of contributions from other columns.
    
    # Alternative: try random linear combinations of columns to see if we
    # can create a zero column (which we can then drop).
    
    rng = np.random.RandomState(42)
    best_rank = rank
    best_U, best_V, best_W = Uc.copy(), Vc.copy(), Wc.copy()
    
    # Strategy: try to find a linear combination of all columns that is zero
    # This means the columns are linearly dependent in the tensor sense.
    
    # More practical: try removing one column at a time and check if the
    # remaining columns still span the same tensor.
    # This is expensive, so use a heuristic:
    # For each pair of columns (a, b), try to see if column b can be
    # expressed as alpha * column_a + (contribution from other columns).
    
    # Simpler heuristic: try random 2x2 "Strassen-like" merges.
    # Pick two columns, form a new column that's their sum, and see if
    # we can adjust other columns to compensate.
    
    # Actually, the most effective approach for small ranks:
    # Try all single-column eliminations with compensation.
    # For column r to be eliminable, we need:
    #   U[:,r] = sum_s c_s * U[:,s]  (for the A-part)
    #   V[:,r] = sum_s c_s * V[:,s]  (for the B-part)  
    #   W[:,r] = sum_s c_s * W[:,s]  (for the C-part)
    # But this is too restrictive. The actual condition is weaker.
    
    # Let's try a different approach: random rank-reduction via
    # finding the rank of the "flattened" tensor.
    
    # The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # We can flatten this and check if the decomposition is minimal.
    
    # For now, try a simple iterative approach:
    # For each column r, try to see if we can absorb it into another column s
    # by adding alpha * (column r) to (column s) and adjusting.
    
    # Given time constraints, let's try a few random perturbations
    # and check if the rank can be reduced.
    
    # Actually, let's try a more systematic approach:
    # Build the "tensor matrix" for each mode and find its rank.
    # If the decomposition has rank r, then each mode flattening has rank <= r.
    # If the actual tensor rank is < r, we can reduce.
    
    # Mode-1 flattening: T_1 is (n*m) x (p*n) matrix where
    # T_1[i*m+j, k*n+i] = sum_r U[i*m+j, r] * W[k*n+i, r] * (sum over k for V)
    # This is getting complex. Let's use a numerical approach.
    
    # Try: for each column r, set it to zero and see if we can recover
    # the missing contributions by scaling other columns.
    
    # Given the 150s budget, let's just try a limited search.
    # Try to find if any column is a linear combination of others
    # in the "tensor product" sense.
    
    # For column r, define the "signature" as the vector of
    # T contributions it makes. If signature_r = sum_s c_s * signature_s,
    # then r is redundant.
    
    # Compute signatures: for column r, sig_r is a (n*m*p)-dim vector
    # where sig_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # But this is the full tensor, which has rank 1 for each column.
    # The sum of all signatures must equal the multiplication tensor.
    
    # A column r is redundant if sig_r = sum_{s != r} c_s * sig_s
    # for some coefficients c_s. But sig_s are rank-1 tensors, so this
    # is a system of equations.
    
    # Let's just try: for each r, solve for c_s such that
    # U[:,r] = sum c_s U[:,s], V[:,r] = sum c_s V[:,s], W[:,r] = sum c_s W[:,s]
    # This is sufficient but not necessary. If it works, we can eliminate r.
    
    for r in range(rank - 1, -1, -1):
        if time.time() - start > max_time:
            break
        # Try to express column r as linear combo of columns 0..r-1
        # Using least squares on all three matrices simultaneously
        Uo = Uc[:, :r]  # (nm, r)
        Vo = Vc[:, :r]  # (mp, r)
        Wo = Wc[:, :r]  # (pn, r)
        Ur = Uc[:, r]
        Vr = Vc[:, r]
        Wr = Wc[:, r]
        
        # Solve min ||Uo*c - Ur||^2 + ||Vo*c - Vr||^2 + ||Wo*c - Wr||^2
        # Stack into one system
        # This is a least squares problem with r unknowns
        # A = [Uo; Vo; Wo], b = [Ur; Vr; Wr]
        # But dimensions differ, so we need to be careful.
        # Actually, we want c such that Uo@c ≈ Ur AND Vo@c ≈ Vr AND Wo@c ≈ Wr
        # Combined residual: ||[Uo;Vo;Wo]@c - [Ur;Vr;Wr]||^2
        
        A_stack = np.vstack([Uo, Vo, Wo])  # ((nm+mp+pn), r)
        b_stack = np.concatenate([Ur, Vr, Wr])
        
        if r == 0:
            continue
        
        try:
            c, residuals, rank_ls, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
            # Check if the fit is good
            pred = A_stack @ c
            resid = np.linalg.norm(b_stack - pred)
            norm_b = np.linalg.norm(b_stack)
            if norm_b > 1e-10 and resid / norm_b < 1e-8:
                # Column r is (approximately) a linear combination of others
                # Eliminate it
                Uc = np.delete(Uc, r, axis=1)
                Vc = np.delete(Vc, r, axis=1)
                Wc = np.delete(Wc, r, axis=1)
                best_rank = Uc.shape[1]
                best_U, best_V, best_W = Uc.copy(), Vc.copy(), Wc.copy()
                # Restart the loop since indices changed
                r = Uc.shape[1] - 1
                continue
        except Exception:
            pass
    
    # Verify the reduced decomposition
    if best_rank < rank and _verify_fast(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W, best_rank
    
    return None, None, None, rank

def _try_pairwise_merge(U, V, W, n, m, p, max_time):
    """
    Try to merge pairs of columns using Strassen-like tricks.
    For two columns r, s, try to find a single column that replaces both
    by exploiting linearity.
    """
    start = time.time()
    rank = U.shape[1]
    rng = np.random.RandomState(77)
    
    # For each pair (r, s), try to see if we can replace them with one column
    # This requires finding U', V', W' such that the combined contribution
    # of r and s equals the contribution of the new column.
    # This is generally not possible without changing other columns.
    
    # A more practical approach: try to find a linear transformation
    # of the column space that reduces the effective rank.
    
    # Given time, let's try random column eliminations with compensation.
    # For column r, try to absorb it into a random other column s by
    # adding alpha * col_r to col_s and checking if the tensor is preserved.
    
    Uc, Vc, Wc = U.copy(), V.copy(), W.copy()
    best_rank = rank
    best_U, best_V, best_W = Uc.copy(), Vc.copy(), Wc.copy()
    
    # Try all single eliminations with least-squares compensation
    for r in range(rank):
        if time.time() - start > max_time:
            break
        others = [x for x in range(rank) if x != r]
        if len(others) == 0:
            break
        
        Uo = Uc[:, others]
        Vo = Vc[:, others]
        Wo = Wc[:, others]
        Ur = Uc[:, r]
        Vr = Vc[:, r]
        Wr = Wc[:, r]
        
        A_stack = np.vstack([Uo, Vo, Wo])
        b_stack = np.concatenate([Ur, Vr, Wr])
        
        try:
            c, _, _, _ = np.linalg.lstsq(A_stack, b_stack, rcond=None)
            pred = A_stack @ c
            resid = np.linalg.norm(b_stack - pred)
            norm_b = np.linalg.norm(b_stack)
            if norm_b > 1e-10 and resid / norm_b < 1e-8:
                # Can eliminate r
                new_U = np.delete(Uc, r, axis=1)
                new_V = np.delete(Vc, r, axis=1)
                new_W = np.delete(Wc, r, axis=1)
                if _verify_fast(new_U, new_V, new_W, n, m, p):
                    best_rank = new_U.shape[1]
                    best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
                    Uc, Vc, Wc = new_U, new_V, new_W
                    # Restart
                    r = 0
                    continue
        except Exception:
            pass
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None, None, None, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin
    
    # Build the rank-81 Strassen-tiled decomposition
    U, V, W = _build_strassen_tiled(n, m, p)
    rank = U.shape[1]
    
    # Verify baseline
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try rank reduction
    if time.time() < deadline - 10:
        t_remaining = deadline - time.time() - 5
        new_U, new_V, new_W, new_rank = _try_reduce(U, V, W, n, m, p, t_remaining)
        if new_U is not None and new_rank < best_rank:
            best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
    
    # Try pairwise merges if time remains
    if time.time() < deadline - 10:
        t_remaining = deadline - time.time() - 5
        new_U, new_V, new_W, new_rank = _try_pairwise_merge(best_U, best_V, best_W, n, m, p, t_remaining)
        if new_U is not None and new_rank < best_rank:
            best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
# EVOLVE-BLOCK-END
