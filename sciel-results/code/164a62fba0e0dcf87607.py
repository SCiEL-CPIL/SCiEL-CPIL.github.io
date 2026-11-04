"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for smaller blocks.
Then applies aggressive peeling with vectorized checks to remove redundant rank-1 terms.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222_local(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block. Rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block. Rank 23 (3x7 Strassen + 2 schoolbook).
        """
        strassen_222_local(i0, j0, k0)
        strassen_222_local(i0, j0, k0 + 2)
        strassen_222_local(i0, j0, k0 + 4)
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m347_decomp():
        """
        Full 3x4 * 4x7 decomposition.
        Split n=3 into 1+2.
        n=1 part: 1x4 * 4x7. Split m=4 into 2+2. Two 1x2x7 blocks.
        n=2 part: 2x4 * 4x7. Split m=4 into 2+2. Two 2x2x7 blocks.
        """
        m127_decomp(0, 0, 0)
        m127_decomp(0, 2, 0)
        m227_decomp(1, 0, 0)
        m227_decomp(1, 2, 0)

    m347_decomp()

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def fast_peel(U_mat, V_mat, W_mat):
        """
        Aggressive peeling: repeatedly check each term for redundancy using
        vectorized numpy operations.
        """
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        rank = U_curr.shape[1]
        
        # Precompute index arrays for vectorized computation
        ijs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        def check_redundant(r, U_c, V_c, W_c, removed_set):
            """Check if term r is redundant given already removed terms."""
            for (i, j, k) in ijs:
                u_row = U_c[i*m+j, :]
                v_row = V_c[j*p+k, :]
                w_row = W_c[k*n+i, :]
                prod = u_row * v_row * w_row
                total = prod.sum() - prod[r]
                for r2 in removed_set:
                    total -= prod[r2]
                if abs(total - 1.0) > 1e-9:
                    return False
            return True
        
        removed = set()
        any_removed = True
        while any_removed and time.time() < deadline - 15:
            any_removed = False
            rank = U_curr.shape[1]
            for r in range(rank):
                if r in removed:
                    continue
                if check_redundant(r, U_curr, V_curr, W_curr, removed):
                    removed.add(r)
                    any_removed = True
        
        if not removed:
            return U_mat, V_mat, W_mat
        
        keep = [r for r in range(rank) if r not in removed]
        return U_curr[:, keep], V_curr[:, keep], W_curr[:, keep]

    def peel_with_local_opt(U_mat, V_mat, W_mat):
        """
        Try to reduce rank by locally optimizing: for each term r, try to
        eliminate it by adjusting one other term s. This works when the
        contribution of r can be absorbed into s.
        """
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        rank = U_curr.shape[1]
        
        ijs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        def try_eliminate_with_adjustment(r, s, U_c, V_c, W_c):
            """Try to eliminate term r by adjusting term s."""
            # For each (i,j,k), the contribution from r is:
            #   U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
            # We want to add a correction to term s such that:
            #   U_c[i*m+j, s] * V_c[j*p+k, s] * W_c[k*n+i, s] + delta = 
            #   U_c[i*m+j, s] * V_c[j*p+k, s] * W_c[k*n+i, s] + 
            #   U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
            # This requires the ratio to be constant across all (i,j,k)
            # where both terms have non-zero contribution.
            
            # Collect all (i,j,k) where term r has non-zero contribution
            contributions = []
            for (i, j, k) in ijs:
                u_r = U_c[i*m+j, r]
                v_r = V_c[j*p+k, r]
                w_r = W_c[k*n+i, r]
                u_s = U_c[i*m+j, s]
                v_s = V_c[j*p+k, s]
                w_s = W_c[k*n+i, s]
                
                contrib_r = u_r * v_r * w_r
                contrib_s = u_s * v_s * w_s
                
                if abs(contrib_r) > 1e-12:
                    if abs(contrib_s) < 1e-12:
                        # Can't absorb if s has zero contribution here
                        return None
                    # The required scale factor for s
                    # We need: new_s_contrib = contrib_s + contrib_r
                    # new_s_contrib / contrib_s = 1 + contrib_r/contrib_s
                    # But this must be achievable by scaling one of u_s, v_s, w_s
                    # Let's check if contrib_r/contrib_s is constant
                    ratio = contrib_r / contrib_s
                    contributions.append((i, j, k, ratio))
            
            if not contributions:
                return None
            
            # Check if all ratios are the same (within tolerance)
            base_ratio = contributions[0][3]
            for _, _, _, ratio in contributions[1:]:
                if abs(ratio - base_ratio) > 1e-9:
                    return None
            
            # If ratios are constant, we can scale one component of term s
            # Let's try scaling U_s
            # new_U_s[i*m+j] = U_s[i*m+j] * (1 + base_ratio) for all relevant (i,j)
            # But this only works if the scaling is consistent
            # Actually, we need: for each (i,j,k) with contrib_r != 0:
            #   U_new_s[i*m+j] * V_s[j*p+k] * W_s[k*n+i] = 
            #   U_s[i*m+j] * V_s[j*p+k] * W_s[k*n+i] + 
            #   U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
            # Divide by V_s[j*p+k] * W_s[k*n+i]:
            #   U_new_s[i*m+j] = U_s[i*m+j] + U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] / (V_s[j*p+k] * W_s[k*n+i])
            # The right side must be independent of k for fixed (i,j)
            
            # Check consistency: for fixed (i,j), the correction must be same for all k
            from collections import defaultdict
            corrections_u = defaultdict(set)
            corrections_v = defaultdict(set)
            corrections_w = defaultdict(set)
            
            for (i, j, k, _) in contributions:
                u_r = U_c[i*m+j, r]
                v_r = V_c[j*p+k, r]
                w_r = W_c[k*n+i, r]
                v_s = V_c[j*p+k, s]
                w_s = W_c[k*n+i, s]
                u_s = U_c[i*m+j, s]
                
                if abs(v_s * w_s) > 1e-12:
                    # Correction to U_s[i*m+j]
                    corr_u = u_r * v_r * w_r / (v_s * w_s)
                    corrections_u[(i, j)].add(corr_u)
                
                if abs(u_s * w_s) > 1e-12:
                    # Correction to V_s[j*p+k]
                    corr_v = u_r * v_r * w_r / (u_s * w_s)
                    corrections_v[(j, k)].add(corr_v)
                
                if abs(u_s * v_s) > 1e-12:
                    # Correction to W_s[k*n+i]
                    corr_w = u_r * v_r * w_r / (u_s * v_s)
                    corrections_w[(k, i)].add(corr_w)
            
            # Check if U correction is consistent
            u_consistent = all(len(vals) <= 1 for vals in corrections_u.values())
            if u_consistent:
                U_new = U_c.copy()
                for (i, j), vals in corrections_u.items():
                    if vals:
                        U_new[i*m+j, s] += list(vals)[0]
                # Verify
                if verify(U_new, V_c, W_c):
                    return (U_new, V_c, W_c)
            
            # Check if V correction is consistent
            v_consistent = all(len(vals) <= 1 for vals in corrections_v.values())
            if v_consistent:
                V_new = V_c.copy()
                for (j, k), vals in corrections_v.items():
                    if vals:
                        V_new[j*p+k, s] += list(vals)[0]
                if verify(U_c, V_new, W_c):
                    return (U_c, V_new, W_c)
            
            # Check if W correction is consistent
            w_consistent = all(len(vals) <= 1 for vals in corrections_w.values())
            if w_consistent:
                W_new = W_c.copy()
                for (k, i), vals in corrections_w.items():
                    if vals:
                        W_new[k*n+i, s] += list(vals)[0]
                if verify(U_c, V_c, W_new):
                    return (U_c, V_c, W_new)
            
            return None
        
        improved = False
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            rank = U_curr.shape[1]
            found = False
            for r in range(rank):
                if time.time() > deadline - 20:
                    break
                for s in range(rank):
                    if s == r:
                        continue
                    result = try_eliminate_with_adjustment(r, s, U_curr, V_curr, W_curr)
                    if result is not None:
                        U_curr, V_curr, W_curr = result
                        # Remove term r
                        keep = [x for x in range(rank) if x != r]
                        U_curr = U_curr[:, keep]
                        V_curr = V_curr[:, keep]
                        W_curr = W_curr[:, keep]
                        found = True
                        improved = True
                        break
                if found:
                    break
            if not found:
                break
        
        return U_curr, V_curr, W_curr

    # Apply peeling iteratively
    U, V, W = fast_peel(U, V, W)
    
    # Second pass
    if time.time() < deadline - 25:
        U2, V2, W2 = fast_peel(U, V, W)
        if U2.shape[1] < U.shape[1]:
            U, V, W = U2, V2, W2
    
    # Try local optimization peeling
    if time.time() < deadline - 30:
        U3, V3, W3 = peel_with_local_opt(U, V, W)
        if U3.shape[1] < U.shape[1]:
            U, V, W = U3, V3, W3
    
    # Another round of fast peeling after local opt
    if time.time() < deadline - 15:
        U4, V4, W4 = fast_peel(U, V, W)
        if U4.shape[1] < U.shape[1]:
            U, V, W = U4, V4, W4
    
    # Final verification
    if not verify(U, V, W):
        # Fall back to original
        U_orig = np.array(U_vecs).T
        V_orig = np.array(V_vecs).T
        W_orig = np.array(W_vecs).T
        return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}

    # Convert to 2D list with exact fractions where possible
    def to_exact_2d_list(arr):
        """Convert numpy 2D array to 2D list, using Fraction for half-integers."""
        result = []
        for row in arr:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                elif abs(2*val - round(2*val)) < 1e-10:
                    row_list.append(Fraction(int(round(2*val)), 2))
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    U_list = to_exact_2d_list(U)
    V_list = to_exact_2d_list(V)
    W_list = to_exact_2d_list(W)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
