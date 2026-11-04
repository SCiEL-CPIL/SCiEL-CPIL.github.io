import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 5
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    def get_strassen_block(i0, j0, k0):
        """Returns 7 term columns for a 2x2x2 Strassen block (global offsets)."""
        terms = []
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        u1 = [((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v1 = [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w1 = [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u1, v1, w1))
        # M2 = (A21+A22)B11 -> C21, C22
        u2 = [((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v2 = [((j0+0)*p + (k0+0), 1)]
        w2 = [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]
        terms.append((u2, v2, w2))
        # M3 = A11(B21-B11) -> C11, C12
        u3 = [((i0+0)*m + (j0+0), 1)]
        v3 = [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)]
        w3 = [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u3, v3, w3))
        # M4 = A22(B11-B22) -> C11, C12
        u4 = [((i0+1)*m + (j0+1), 1)]
        v4 = [((j0+1)*p + (k0+0), 1), ((j0+0)*p + (k0+0), -1)]
        w4 = [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]
        terms.append((u4, v4, w4))
        # M5 = (A11+A12)B22 -> C11, C12
        u5 = [((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)]
        v5 = [((j0+1)*p + (k0+1), 1)]
        w5 = [((k0+0)*n + (i0+0), -1), ((k0+1)*n + (i0+0), 1)]
        terms.append((u5, v5, w5))
        # M6 = (A21-A11)(B11+B12) -> C21
        u6 = [((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)]
        v6 = [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)]
        w6 = [((k0+1)*n + (i0+1), 1)]
        terms.append((u6, v6, w6))
        # M7 = (A12-A22)(B21+B22) -> C21
        u7 = [((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v7 = [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w7 = [((k0+0)*n + (i0+0), 1)]
        terms.append((u7, v7, w7))
        return terms

    def sparse_to_dense(sparse, dim):
        arr = np.zeros(dim)
        for idx, val in sparse:
            arr[idx] = val
        return arr

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        U_cols.append(sparse_to_dense(u_sparse, dim_u))
        V_cols.append(sparse_to_dense(v_sparse, dim_v))
        W_cols.append(sparse_to_dense(w_sparse, dim_w))

    def add_schoolbook(i, j, k):
        u = np.zeros(dim_u); u[i*m + j] = 1
        v = np.zeros(dim_v); v[j*p + k] = 1
        w = np.zeros(dim_w); w[k*n + i] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    covered = set()

    # Four Strassen blocks covering i in {0,1}, all j in {0..3}, all k in {0..3}
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            terms = get_strassen_block(0, j0, k0)
            for u, v, w in terms:
                add_term(u, v, w)
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((0 + di, j0 + dj, k0 + dk))

    # Schoolbook for the rest (i=2, all j, all k) and k=4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                add_schoolbook(i, j, k)

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]

    def verify(U, V, W, tol=1e-9):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    s = float(np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    max_err = verify(U, V, W)
    if max_err > 1e-9:
        # Fallback to schoolbook
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        rank = n*m*p

    # Linear dependency elimination using rational arithmetic
    def find_linear_dependencies(U, V, W, rank):
        """
        Find linear combinations of rank-1 terms that sum to zero.
        We look for vectors c in Q^rank such that for all (i,j,k):
        sum_r c_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
        
        This is a nonlinear constraint, so we use a greedy approach:
        Try removing each term and see if the remaining terms can still
        reproduce the tensor via linear recombination.
        """
        # Precompute contribution tensor C[i,j,k,r]
        C = np.zeros((n, m, p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C[i, j, k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        S = C.sum(axis=3).copy()
        
        # Try to find removable terms by checking if removing a term
        # allows the remaining terms to still sum to 1 via scaling
        removed = set()
        
        # First pass: try removing each term individually
        for r in range(rank):
            if r in removed:
                continue
            if time.time() > deadline - 10:
                break
            # Check if we can scale other terms to compensate
            # This is equivalent to finding if C[:,:, :, r] is in the span
            # of other C[:,:, :, r'] in a multiplicative sense
            # For simplicity, try pairwise: can we remove r if some other r' 
            # has the same support pattern?
            pass
        
        # Better approach: use the fact that Strassen has known identities
        # For each Strassen block, we can try to express one term as a combination
        # of others by solving a small linear system
        
        # Actually, let's use a different strategy: 
        # Try all pairs of terms and see if their difference can be absorbed
        # by adjusting other terms
        
        # Most practical: try removing terms that are "redundant" in the sense
        # that their contribution is already covered by other terms
        # We'll use a greedy approach with verification
        
        kept = list(range(rank))
        
        # Sort by sparsity (fewer nonzeros first) - these are more likely redundant
        col_sparsity = np.array([(C[:, :, :, r] != 0).sum() for r in range(rank)])
        cand_order = sorted(kept, key=lambda r: col_sparsity[r])
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for r in cand_order:
                if r not in kept:
                    continue
                if time.time() > deadline - 10:
                    break
                
                # Try removing r and see if we can rescale others to fix
                # This is hard in general, so let's try a simpler heuristic:
                # Check if r's contribution is a scalar multiple of another term's
                # contribution on their common support
                
                # For now, just try removing and verifying if the sum is still 1
                # with adjusted weights (this is a simplification)
                new_S = S - C[:, :, :, r]
                
                # Check if new_S can be made to equal 1 by scaling remaining terms
                # This requires solving a system, which is expensive
                # Skip for now and rely on the fact that Strassen is already minimal
                # per block in most cases
                pass
        
        return kept

    # Since finding true linear dependencies in the tensor sense is NP-hard,
    # we'll use a practical approach: try to merge schoolbook terms with Strassen terms
    # by finding overlapping contributions
    
    def try_merge_terms(U, V, W, rank, deadline):
        """
        Try to reduce rank by finding terms that can be merged.
        Two rank-1 terms (u1,v1,w1) and (u2,v2,w2) can be merged if
        there exist scalars a,b such that a*(u1,v1,w1) + b*(u2,v2,w2)
        can be replaced by a single term (u',v',w').
        This happens when u1 and u2 are proportional, v1 and v2 are proportional,
        and w1 and w2 are proportional (up to a common scalar).
        """
        kept = list(range(rank))
        
        # Check for proportional terms
        for i in range(len(kept)):
            if time.time() > deadline - 10:
                break
            r1 = kept[i]
            if r1 not in kept:
                continue
            for j in range(i+1, len(kept)):
                r2 = kept[j]
                if r2 not in kept:
                    continue
                
                # Check if U[:,r1] is proportional to U[:,r2]
                u1 = U[:, r1]
                u2 = U[:, r2]
                v1 = V[:, r1]
                v2 = V[:, r2]
                w1 = W[:, r1]
                w2 = W[:, r2]
                
                # Check proportionality
                nonz_u1 = np.nonzero(u1)[0]
                nonz_u2 = np.nonzero(u2)[0]
                if len(nonz_u1) != len(nonz_u2) or not np.array_equal(nonz_u1, nonz_u2):
                    continue
                
                nonz_v1 = np.nonzero(v1)[0]
                nonz_v2 = np.nonzero(v2)[0]
                if len(nonz_v1) != len(nonz_v2) or not np.array_equal(nonz_v1, nonz_v2):
                    continue
                
                nonz_w1 = np.nonzero(w1)[0]
                nonz_w2 = np.nonzero(w2)[0]
                if len(nonz_w1) != len(nonz_w2) or not np.array_equal(nonz_w1, nonz_w2):
                    continue
                
                # Check if ratios are consistent
                try:
                    ratio_u = u1[nonz_u1] / u2[nonz_u1]
                    ratio_v = v1[nonz_v1] / v2[nonz_v1]
                    ratio_w = w1[nonz_w1] / w2[nonz_w1]
                    
                    if not np.allclose(ratio_u, ratio_u[0], atol=1e-9):
                        continue
                    if not np.allclose(ratio_v, ratio_v[0], atol=1e-9):
                        continue
                    if not np.allclose(ratio_w, ratio_w[0], atol=1e-9):
                        continue
                    
                    ru = ratio_u[0]
                    rv = ratio_v[0]
                    rw = ratio_w[0]
                    
                    # If ru == rv == rw, we can merge
                    if np.isclose(ru, rv, atol=1e-9) and np.isclose(rv, rw, atol=1e-9):
                        # Merge: replace r1 and r2 with a single term
                        # The new term is (u1+v1, v1+v2, w1+w2) scaled appropriately
                        # Actually, if they're identical up to scalar, we can just add them
                        # and remove one
                        kept.remove(r2)
                        improved = True
                except (ZeroDivisionError, IndexError):
                    continue
        
        kept_arr = np.array(kept, dtype=int)
        U_new = U[:, kept_arr]
        V_new = V[:, kept_arr]
        W_new = W[:, kept_arr]
        return U_new, V_new, W_new, len(kept)

    if time.time() < deadline - 15:
        U, V, W, rank = try_merge_terms(U, V, W, rank, deadline)
    
    # Final verification
    max_err = verify(U, V, W)
    if max_err > 1e-9:
        # Fallback
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        rank = n*m*p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
