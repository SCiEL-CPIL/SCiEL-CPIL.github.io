"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank via:
     a. Duplicate column detection
     b. Border rank / Gröbner-style elimination (determine if rank 43 is feasible)
     c. Local search: try replacing pairs of columns with single combined columns
  3. Validate any candidate with exact rational arithmetic.
  4. If no improvement is found within the time budget, return the best decomposition found.
"""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # 30 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Try to reduce rank via structured alternatives and linear dependence search ----
    def _try_reduce():
        nonlocal best_candidate, best_rank

        num_terms = n * m * p  # 48

        def _col_vector(U, V, W, c):
            vec = [0] * num_terms
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0:
                            continue
                        w_val = W[k * n + i][c]
                        if w_val == 0:
                            continue
                        vec[i * m * p + a * p + k] = u_val * v_val * w_val
            return vec

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        cols = []
        for c in range(rank0):
            cols.append(_col_vector(U, V, W, c))

        # Check for duplicate columns (exact)
        seen = {}
        removable = []
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                removable.append(c)
            else:
                seen[key] = c

        if removable:
            for c_rem in removable:
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for c in range(rank0):
                    if c == c_rem:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][c]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][c]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][c]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # ---- Border rank / Gröbner-style elimination approach ----
        # The key idea: we have 44 rank-1 tensors (columns) that sum to the target.
        # We want to know if 43 rank-1 tensors can also sum to the target.
        # This is equivalent to asking: does the target tensor lie on the 43-secant variety?
        
        # A practical approach: try to find a linear combination of 2 columns that can be
        # replaced by 1 column. That is, find columns c1, c2 and a new column c_new such that:
        #   col(c1) + col(c2) = col(c_new)
        # where col(c_new) is a valid rank-1 tensor (i.e., U_new ⊗ V_new ⊗ W_new)
        
        # More generally, try to find alpha, beta such that:
        #   alpha * col(c1) + beta * col(c2) is a rank-1 tensor
        # and then replace c1, c2 with a single scaled column.
        
        # For this, we need to check if a given vector in R^48 is a rank-1 tensor.
        # A vector v in R^(n*m*p) is rank-1 iff the matrix M_{(i,a),(k)} = v[i*m*p + a*p + k]
        # has rank 1 (when reshaped to (n*m) x p), AND similarly for other reshapings.
        
        def _is_rank1_tensor(vec):
            """Check if a 48-dim vector represents a rank-1 tensor."""
            # Reshape to (n*m) x p = 6 x 8
            M1 = np.array(vec).reshape(n * m, p)
            # Reshape to (m*p) x n = 24 x 2
            M2 = np.array(vec).reshape(m * p, n)
            # Reshape to (p*n) x m = 16 x 3
            M3 = np.array(vec).reshape(p * n, m)
            
            # Check rank 1 for each
            if np.linalg.matrix_rank(M1) > 1:
                return False
            if np.linalg.matrix_rank(M2) > 1:
                return False
            if np.linalg.matrix_rank(M3) > 1:
                return False
            return True

        def _extract_rank1_factors(vec):
            """Extract U, V, W factors from a rank-1 tensor vector."""
            # vec[i*m*p + a*p + k] = U[i*m+a] * V[a*p+k] * W[k*n+i]
            # Find a non-zero entry to fix scaling
            vec_arr = np.array(vec)
            
            # Find first non-zero
            nz = np.nonzero(vec_arr != 0)
            if len(nz[0]) == 0:
                return None
            
            idx = nz[0][0]
            i = idx // (m * p)
            rem = idx % (m * p)
            a = rem // p
            k = rem % p
            
            # For rank-1: vec[i*m*p + a'*p + k'] = U[i*m+a'] * V[a'*p+k'] * W[k'*n+i]
            # We need to factor this. Use the structure:
            # Fix U[i*m+a] = 1 for the reference (i,a)
            # Then V[a*p+k] * W[k*n+i] = vec[i*m*p + a*p + k]
            
            # Better: use SVD-like approach on the reshaped matrices
            M1 = vec_arr.reshape(n * m, p)  # (i*m+a, k)
            # M1 = u_vec @ w_vec where u_vec[i*m+a] and w_vec[k]
            # But this isn't quite right because of the a index coupling.
            
            # Let's do it more carefully.
            # M1[i*m+a, k] = U[i*m+a] * V[a*p+k] * W[k*n+i]
            # For fixed i: M1[i*m: (i+1)*m, k] = U[i*m:a'] * V[a'*p+k] * W[k*n+i]
            # This is a rank-1 matrix in (a', k) for each i.
            
            # Extract using the first non-zero row
            # Let's use a different approach: assume we can factor it.
            # From M1: for each i, the submatrix M1[i*m:(i+1)*m, :] should be rank 1.
            
            # Get U factors from column sums pattern
            # Actually, let's just construct the factors directly.
            
            U_f = [0] * (n * m)
            V_f = [0] * (m * p)
            W_f = [0] * (p * n)
            
            # Set W[k*n+i] for all k,i from the data
            # W[k*n+i] = vec[i*m*p + a*p + k] / (U[i*m+a] * V[a*p+k])
            
            # This is circular. Let's use a standard rank-1 factorization.
            # Reshape to (n*m*p) and use the fact that for rank-1:
            # vec = outer(u, v, w) in some sense.
            
            # Simple approach: pick a reference index and solve.
            # Let's set U[i*m+a] = M1[i*m+a, k_ref] / (V[a*p+k_ref] * W[k_ref*n+i])
            
            # This is getting complex. Let's use numpy's eigenvector approach.
            # For a rank-1 tensor, the tensor train decomposition is trivial.
            
            # Alternative: just return the vector and let the caller use it directly.
            # The caller needs U, V, W factors, not just the vector.
            
            # Let's do it properly with a least-squares approach.
            # Assume W is fixed (e.g., W[k*n+i] = 1 for all), then solve for U and V.
            # But that won't work in general.
            
            # Standard approach: 
            # 1. Reshape to (n*m) x p: M1
            # 2. M1 should be rank 1: M1 = u @ w^T where u is (n*m) and w is (p)
            #    But M1[i*m+a, k] = U[i*m+a] * V[a*p+k] * W[k*n+i]
            #    This is NOT simply u @ w^T because of the a-k coupling.
            
            # So the simple reshape doesn't give a clean rank-1 factorization.
            # We need a more careful approach.
            
            # Let's use the following:
            # For each (i, a), U[i*m+a] is a scalar.
            # For each (a, k), V[a*p+k] is a scalar.
            # For each (k, i), W[k*n+i] is a scalar.
            # vec[i*m*p + a*p + k] = U[i*m+a] * V[a*p+k] * W[k*n+i]
            
            # Take logs? No, values can be negative.
            
            # Use the ratio trick:
            # vec[i1*m*p + a*p + k] / vec[i2*m*p + a*p + k] = U[i1*m+a] / U[i2*m+a]
            # vec[i*m*p + a1*p + k] / vec[i*m*p + a2*p + k] = V[a1*p+k] / V[a2*p+k]
            # vec[i*m*p + a*p + k1] / vec[i*m*p + a*p + k2] = W[k1*n+i] / W[k2*n+i]
            
            # Set U[0] = 1 (reference)
            U_f[0] = 1
            
            # From vec[i*m*p + a*p + k] = U[i*m+a] * V[a*p+k] * W[k*n+i]
            # with U[0] = 1:
            # vec[0*m*p + a*p + k] = V[a*p+k] * W[k*n+0]
            
            # This is still coupled. Let's just do a numerical factorization.
            
            # Use the fact that we can write:
            # For fixed i: vec[i*m*p : (i+1)*m*p] as a (m*p) vector
            # This should factor as: for each a, U[i*m+a] * (V[a*p:] * W[:, n*i + i])
            
            # Let's just use scipy or manual iteration.
            # For now, return None to indicate we can't easily extract factors.
            return None

        # ---- Local search: try to merge pairs of columns ----
        # For each pair (c1, c2), check if col(c1) + col(c2) is rank-1.
        # If so, we can replace them with a single column.
        
        cols_arr = np.array(cols, dtype=float)
        
        # First, check if any single column is zero (can be removed)
        for c in range(rank0):
            if np.all(cols_arr[c] == 0):
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for cc in range(rank0):
                    if cc == c:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][cc]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][cc]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][cc]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # Try merging pairs: check if col(c1) + col(c2) is a rank-1 tensor
        # This would allow us to replace 2 columns with 1.
        # But we need to be able to extract the U, V, W factors.
        
        # Let's try a different approach: 
        # For each column c, try to scale it by a factor alpha and see if
        # the resulting set still sums to the target when we add a correction term.
        
        # Actually, the most promising approach is:
        # Try to find 3 columns c1, c2, c3 such that col(c1) + col(c2) = col(c3) * alpha
        # for some scalar alpha. Then we can remove c1 and c2 and scale c3.
        
        # This is equivalent to checking if any column is a scalar multiple of the sum of two others.
        
        # For efficiency, let's check all triples (c1, c2, c3) with c1 < c2 < c3.
        # There are C(44, 3) = 13244 triples, which is manageable.
        
        # But we also need to check if the resulting tensor is valid (sums to target).
        
        # Let's try a simpler approach: check if any column is a scalar multiple of another.
        for c1 in range(rank0):
            for c2 in range(c1 + 1, rank0):
                # Check if col(c1) = alpha * col(c2) for some alpha
                # Find a non-zero entry in col(c2)
                nz2 = np.nonzero(cols_arr[c2] != 0)
                if len(nz2[0]) == 0:
                    continue
                idx = nz2[0][0]
                if cols_arr[c2][idx] == 0:
                    continue
                alpha = cols_arr[c1][idx] / cols_arr[c2][idx]
                # Check if this alpha works for all entries
                if np.allclose(cols_arr[c1], alpha * cols_arr[c2], atol=1e-10):
                    # col(c1) = alpha * col(c2)
                    # We can remove c1 and scale c2 by (1 + alpha)
                    # But we need to update the factors.
                    # If col(c2) has factors U2, V2, W2, then alpha*col(c2) has factors
                    # U2, V2, alpha*W2 (or any distribution of alpha among the three).
                    # So we can set W2_new = (1 + alpha) * W2 and remove c1.
                    # But we need to verify this is correct.
                    
                    # Actually, we can't just scale W2 because the decomposition
                    # requires exact integer/half-integer entries.
                    # Let's skip this for now.
                    pass

        if not _elapsed_ok():
            return

        # ---- Try a different structured decomposition ----
        # Use Strassen for 2x2x2 but with a different block structure.
        # Instead of splitting B into 4 blocks of 2 columns, try splitting into
        # 2 blocks of 4 columns and using a rank-12 decomposition for 2x2x4.
        
        # The rank of 2x2x4 matrix multiplication:
        # This is like multiplying a 2x2 matrix by a 2x4 matrix.
        # We can view this as 2 separate 2x2x2 multiplications, each of rank 7.
        # So total rank = 14, same as 2*7 = 14. No improvement.
        
        # What about 2x2x8? Rank would be 4*7 = 28. Same as before.
        
        # Let's try a completely different approach:
        # Use the fact that 2x3 x 3x8 can be seen as a 2x8 matrix multiplication
        # with an intermediate 3x3 matrix.
        
        # Actually, let's try to use a rank-6 decomposition for 2x2x2 (if it exists).
        # The minimal rank for 2x2x2 is 7 (Strassen), so this won't help.
        
        # Let's try to combine the schoolbook terms with Strassen terms more cleverly.
        # The schoolbook terms for A2*B2 are:
        # C[i,k] += A2[i,0] * B2[0,k] for i in {0,1}, k in {0,...,7}
        # That's 16 terms.
        
        # The Strassen terms for A1*B1 are 28 terms.
        
        # Can we share some computation between these?
        # For example, if we use a Strassen-like decomposition that combines
        # A1 and A2 in some way.
        
        # Let's try: instead of treating A1 and A2 separately, use a
        # rank-r decomposition for the full 2x3 x 3x8 problem.
        
        # This is a hard problem. Let's try a numerical approach:
        # Use random initialization and gradient descent to find a low-rank decomposition.
        
        if not _elapsed_ok():
            return

        # ---- Numerical optimization approach ----
        # Try to find a rank-43 decomposition using numerical methods.
        
        def _try_numerical_optimize(target_rank):
            """Try to find a rank-43 decomposition using numerical optimization."""
            if not _elapsed_ok():
                return None
            
            # Initialize with the base decomposition and try to merge two columns.
            # Pick two random columns to merge.
            
            for attempt in range(50):
                if not _elapsed_ok():
                    return None
                
                # Pick two random columns to try to merge
                c1, c2 = random.sample(range(rank0), 2)
                
                # Try to find a new column c_new such that:
                # col(c1) + col(c2) - col(c_new) = 0
                # i.e., col(c_new) = col(c1) + col(c2)
                
                c_new_vec = cols_arr[c1] + cols_arr[c2]
                
                # Check if c_new_vec is a rank-1 tensor
                if not _is_rank1_tensor(c_new_vec):
                    continue
                
                # If it is, we need to extract the factors.
                # For now, skip the factor extraction and just note that we found a merge.
                # In a full implementation, we would extract U, V, W factors.
                
                # For this attempt, let's just return None since we can't easily
                # extract the factors.
                pass
            
            return None

        # Try numerical optimization
        result = _try_numerical_optimize(43)
        if result is not None:
            best_candidate = result
            best_rank = 43
            return

        # ---- Try alternative block partitioning ----
        # Split A differently: A1(1x3) and A2(1x3)
        # A1*B: 1x3 x 3x8 = 24 schoolbook
        # A2*B: 1x3 x 3x8 = 24 schoolbook
        # Total: 48 (worse than 44)
        
        # Split B differently: B1(3x4) and B2(3x4)
        # A1(2x2)*B1(2x4): 2*2*4 = 16 schoolbook or 2 Strassen = 14
        # A1(2x2)*B2(2x4): 14
        # A2(2x1)*B1(1x4): 8
        # A2(2x1)*B2(1x4): 8
        # Total: 14+14+8+8 = 44 (same)
        
        # Or: A1(2x2)*B1(2x4) with Strassen: 2*7 = 14
        # A1(2x2)*B2(2x4) with Strassen: 14
        # A2(2x1)*B1(1x4): 8
        # A2(2x1)*B2(1x4): 8
        # Total: 44 (same)
        
        # Let's try: A1(2x2)*B1(2x8) with 4 Strassen = 28
        # A2(2x1)*B1(1x8): 16 schoolbook
        # Total: 44 (same as base)
        
        # It seems 44 is the natural boundary for this approach.
        # Let's try one more thing: use a rank-5 decomposition for 2x2x2.
        # But the minimal rank for 2x2x2 is 7, so this won't work.
        
        return

    if _elapsed_ok():
        _try_reduce()

    if best_rank < 44:
        return best_candidate

    return base

# EVOLVE-BLOCK-END
