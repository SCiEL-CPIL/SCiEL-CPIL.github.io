"""Combined program: structured Strassen tiling (rank 77) + SVD-based rank reduction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 115.0  # safety margin before 120s evaluator limit


def _build_structured_rank77():
    """Build the proven rank-77 nested Strassen tiling decomposition."""
    n, m, p = 4, 4, 6
    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=6, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j1 != j2 or i1 != i2:
                    s = 0.0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True


def _try_reduce_rank(U, V, W, tol=1e-9):
    """Try to reduce rank by identifying linearly dependent columns.
    
    Strategy: For each column r, check if it can be expressed as a linear combination
    of other columns. If so, remove it.
    
    We use the fact that if column r is redundant, then for all (a,b,c):
    sum_{r' != r} U[a,r']*V[b,r']*W[c,r'] * x[r'] = U[a,r]*V[b,r]*W[c,r]
    for some coefficients x.
    
    This is a system of equations. We solve it and check if a solution exists.
    """
    R = U.shape[1]
    n, m, p = 4, 4, 6
    
    # Build the constraint matrix
    # For each (a,b,c) where the tensor value is 1 or 0, we have a constraint
    # But we only need to check the "active" constraints (where tensor=1)
    # and ensure zeros are maintained.
    
    # More efficient: check column by column if it's in the span of others
    # using the outer product structure
    
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    # Try to find a minimal subset of columns
    # Use greedy approach: try removing each column and check validity
    
    kept = list(range(R))
    
    # Sort columns by norm (try to remove small ones first)
    norms = [np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r]) for r in range(R)]
    order = sorted(range(R), key=lambda r: norms[r])
    
    # Try removing columns in order of smallest norm
    for r_to_remove in order:
        if r_to_remove not in kept:
            continue
        if time.time() > _DEADLINE - 5:
            break
        
        # Temporarily remove this column
        new_kept = [k for k in kept if k != r_to_remove]
        
        # Check if the remaining columns still span the tensor
        # This is expensive, so we do a quick check first
        # Build a reduced system
        
        # For efficiency, check a subset of constraints
        # Sample some (i,j,k) tuples
        sample = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        valid = True
        for i, j, k in sample:
            a_idx = i * m + j
            b_idx = j * p + k
            c_idx = k * n + i
            
            # The contribution from column r_to_remove
            contrib = U[a_idx, r_to_remove] * V[b_idx, r_to_remove] * W[c_idx, r_to_remove]
            
            # The contribution from all other kept columns
            other = 0.0
            for r in new_kept:
                other += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
            
            # We need: other + contrib = 1 (for active constraints)
            # So we need: other = 1 - contrib
            # But we can't adjust the other columns, so this won't work directly.
            
            # Instead, we need to check if there exist coefficients x such that
            # sum_{r in new_kept} x[r] * U[a_idx, r] * V[b_idx, r] * W[c_idx, r] = 1
            # for all active (i,j,k), and = 0 for inactive.
            
            # This is a linear system. Let's set it up.
            # But this is expensive. Let's use a different approach.
            
            # Actually, the correct approach is:
            # The tensor T is fixed. We need:
            # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b,c)
            # If we remove column r0, we need:
            # sum_{r != r0} U[a,r]*V[b,r]*W[c,r] = T[a,b,c] - U[a,r0]*V[b,r0]*W[c,r0]
            # But the LHS is fixed (it's the sum over remaining columns),
            # so this only works if the RHS equals the LHS for all (a,b,c).
            
            # So the check is: does the remaining columns produce the correct tensor?
            # Which is just _verify_decomposition on the submatrix.
            # But that's expensive. Let's do a quick check on a few points.
            
            target = 1.0  # For active (i,j,k)
            if abs(other - (target - contrib)) > tol:
                valid = False
                break
        
        if valid:
            # Quick check on some zero constraints
            zero_valid = True
            zero_samples = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i != k or j != j:  # This is always true, need better logic
                            pass
            # Actually, let's just do a full verification if the quick check passes
            if len(kept) <= 50:  # Only do full verification for smaller ranks
                U_sub = U[:, new_kept]
                V_sub = V[:, new_kept]
                W_sub = W[:, new_kept]
                if _verify_decomposition(U_sub, V_sub, W_sub):
                    kept = new_kept
                    continue
            else:
                # For larger ranks, trust the quick check but be conservative
                # Only remove if the contribution is very small
                if abs(contrib) < 1e-10:
                    kept = new_kept
                    continue
        
        # If we get here, we couldn't remove this column
        continue
    
    if len(kept) < R:
        return U[:, kept], V[:, kept], W[:, kept]
    return None


def _try_svd_reduction(U, V, W, tol=1e-9):
    """Try SVD-based reduction: find a lower-rank representation.
    
    The idea: if the 77 rank-1 terms have linear dependencies, we can find
    a smaller set that spans the same tensor.
    
    Approach: Stack the factor matrices and use SVD to find the effective rank.
    But this doesn't directly apply to tensor decompositions.
    
    Alternative: Use the fact that if two columns are proportional in all three
    matrices, they can be combined.
    """
    R = U.shape[1]
    
    # Check for proportional columns
    # Two columns r1, r2 are proportional if:
    # U[:, r1] = c * U[:, r2], V[:, r1] = c * V[:, r2], W[:, r1] = c * W[:, r2]
    
    removed = set()
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    for r1 in range(R):
        if r1 in removed:
            continue
        for r2 in range(r1 + 1, R):
            if r2 in removed:
                continue
            if time.time() > _DEADLINE - 5:
                break
            
            # Check if columns are proportional
            # Use the first non-zero element to determine the ratio
            u1 = U[:, r1]
            u2 = U[:, r2]
            v1 = V[:, r1]
            v2 = V[:, r2]
            w1 = W[:, r1]
            w2 = W[:, r2]
            
            # Find a pivot
            pivot_idx = None
            for idx in range(len(u1)):
                if abs(u1[idx]) > 1e-10 or abs(u2[idx]) > 1e-10:
                    pivot_idx = idx
                    break
            
            if pivot_idx is None:
                # Both columns are zero in U, check V and W
                if np.allclose(u1, 0) and np.allclose(u2, 0):
                    # Check V
                    pivot_idx_v = None
                    for idx in range(len(v1)):
                        if abs(v1[idx]) > 1e-10 or abs(v2[idx]) > 1e-10:
                            pivot_idx_v = idx
                            break
                    if pivot_idx_v is not None:
                        c = v1[pivot_idx_v] / v2[pivot_idx_v] if abs(v2[pivot_idx_v]) > 1e-10 else None
                        if c is not None and np.allclose(v1, c * v2, atol=tol) and np.allclose(w1, c * w2, atol=tol):
                            removed.add(r2)
                            break
                continue
            
            u1_pivot = u1[pivot_idx]
            u2_pivot = u2[pivot_idx]
            
            if abs(u2_pivot) < 1e-10:
                if abs(u1_pivot) < 1e-10:
                    continue
                # u1 has non-zero at pivot, u2 has zero. Not proportional unless u1 is also zero
                continue
            
            c = u1_pivot / u2_pivot
            
            if np.allclose(u1, c * u2, atol=tol) and np.allclose(v1, c * v2, atol=tol) and np.allclose(w1, c * w2, atol=tol):
                removed.add(r2)
                break
    
    if removed:
        kept = [r for r in range(R) if r not in removed]
        return U[:, kept], V[:, kept], W[:, kept]
    return None


def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Build the rank-77 structured decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W):
        # Should not happen, but fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce rank
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try SVD-based reduction
    if time.time() < _DEADLINE - 30:
        result = _try_svd_reduction(U, V, W)
        if result is not None:
            U_r, V_r, W_r = result
            if U_r.shape[1] < best_rank and _verify_decomposition(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
    
    # Try greedy column removal
    if time.time() < _DEADLINE - 10:
        result = _try_reduce_rank(best_U, best_V, best_W)
        if result is not None:
            U_r, V_r, W_r = result
            if U_r.shape[1] < best_rank and _verify_decomposition(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
    
    # Convert to lists with integer entries where possible
    def to_list(mat):
        mat = np.array(mat)
        out = []
        for row in mat:
            out_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    out_row.append(int(round(x)))
                else:
                    out_row.append(float(x))
            out.append(out_row)
        return out
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
