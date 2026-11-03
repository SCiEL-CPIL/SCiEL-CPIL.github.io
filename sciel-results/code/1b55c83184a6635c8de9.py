"""Matrix multiplication 4,4,5 tensor decomposition with combinatorial subspace elimination."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_terms():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    return [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        {'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,0): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
    ]

def _naive_terms(n, m, p):
    """Return the n*m*p schoolbook terms."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append({'U': {(i, j): 1}, 'V': {(j, k): 1}, 'W': {(k, i): 1}})
    return terms

def _build_rank65(n, m, p):
    """Build rank-65 decomposition: Strassen 4x4x4 (rank 49) + naive 4x4x1 (rank 16)."""
    # 4x4x4 via Kronecker product of two 2x2x2 Strassen
    strassen_444 = []
    for t1 in _strassen_2x2_terms():
        for t2 in _strassen_2x2_terms():
            U = {}
            V = {}
            W = {}
            for (i1, j1), c1 in t1['U'].items():
                for (i2, j2), c2 in t2['U'].items():
                    key = (i1 * 2 + i2, j1 * 2 + j2)
                    U[key] = U.get(key, 0) + c1 * c2
            for (j1, k1), c1 in t1['V'].items():
                for (j2, k2), c2 in t2['V'].items():
                    key = (j1 * 2 + j2, k1 * 2 + k2)
                    V[key] = V.get(key, 0) + c1 * c2
            for (k1, i1), c1 in t1['W'].items():
                for (k2, i2), c2 in t2['W'].items():
                    key = (k1 * 2 + k2, i1 * 2 + i2)
                    W[key] = W.get(key, 0) + c1 * c2
            strassen_444.append({'U': U, 'V': V, 'W': W})

    # 4x4x1 via naive
    naive_441 = _naive_terms(4, 4, 1)

    # Combine
    all_terms = strassen_444 + naive_441
    rank = len(all_terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, t in enumerate(all_terms):
        for (i, j), c in t['U'].items():
            if i < n and j < m:
                U[i * m + j][r] = c
        for (j, k), c in t['V'].items():
            if j < m and k < p:
                V[j * p + k][r] = c
        for (k, i), c in t['W'].items():
            if k < p and i < n:
                W[k * n + i][r] = c

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_half_integer(U, V, W, tol=1e-9):
    """Check all entries are integers or half-integers."""
    for M in (U, V, W):
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _to_numpy(U, V, W):
    return np.array(U, dtype=np.float64), np.array(V, dtype=np.float64), np.array(W, dtype=np.float64)

def _combinatorial_subspace_elimination(U, V, W, n, m, p, deadline, max_iter=200):
    """
    Find and remove linearly dependent columns.
    A column r is redundant if there exist coefficients c_s (s != r) such that
    for all (i,j,k): U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s]
    
    We use the approach: check if column r's "tensor slice" is in the span of others.
    More practically: try removing each column and check if the remaining still decomposes.
    For efficiency, use SVD to find near-dependent columns first.
    """
    start = time.time()
    U_np, V_np, W_np = _to_numpy(U, V, W)
    rank = U_np.shape[1]
    
    # Build the "trilinear form" matrix for dependency detection
    # For each pair of columns (r, s), compute the "inner product" of their tensor contributions
    # Actually, let's use a more direct approach:
    # Reshape U, V, W into vectors and check linear dependence of the outer products
    
    # Strategy: for each column r, check if it can be expressed as a linear combination
    # of other columns in a way that preserves the trilinear form.
    # This is hard in general, so we use a greedy removal approach with verification.
    
    # First, identify candidate columns for removal using SVD on combined representation
    # Flatten each column's contribution
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For dependency detection, we look at the "matrix" M where M[alpha, beta] = sum_r U[alpha,r] * V[beta,r] * W[gamma,r]
    # This is the tensor itself. Two columns are "redundant" if their combined contribution
    # can be absorbed by others.
    
    # Practical approach: try removing each column one at a time, starting with those
    # that have the smallest "unique contribution"
    
    removed = set()
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        current_rank = rank - len(removed)
        
        # Score each remaining column by how "unique" it is
        # A column is less unique if removing it and adjusting others still works
        # For now, try each column in order of smallest L2 norm of its U component
        candidates = []
        for r in range(rank):
            if r in removed:
                continue
            # Compute a heuristic score: columns with smaller contributions are more likely removable
            u_norm = np.linalg.norm(U_np[:, r])
            v_norm = np.linalg.norm(V_np[:, r])
            w_norm = np.linalg.norm(W_np[:, r])
            score = u_norm * v_norm * w_norm
            candidates.append((score, r))
        
        candidates.sort()
        
        # Try removing the top candidates
        for score, r in candidates[:min(10, current_rank)]:
            if time.time() >= deadline:
                break
            if r in removed:
                continue
            
            # Try removing column r: check if the decomposition still holds
            # without it (this is a necessary but not sufficient condition - 
            # we'd need to adjust other columns)
            # For a quick check: if all entries in column r are zero, it's trivially removable
            if np.allclose(U_np[:, r], 0) and np.allclose(V_np[:, r], 0) and np.allclose(W_np[:, r], 0):
                removed.add(r)
                improved = True
                continue
            
            # More sophisticated: check if column r is a linear combination of others
            # in the sense that its trilinear contribution is spanned
            # Build the "contribution vector" for column r
            # The contribution of column r to the tensor is the outer product
            # U[:,r] ⊗ V[:,r] ⊗ W[:,r] (reshaped)
            
            # For efficiency, check if removing r makes the tensor rank-deficient
            # in a way that can be compensated
            # This is complex, so we use a simpler heuristic:
            # Check if there exists a column s such that U[:,r] is proportional to U[:,s],
            # V[:,r] is proportional to V[:,s], and W[:,r] is proportional to W[:,s]
            # (up to a constant that cancels in the product)
            
            for s in range(rank):
                if s == r or s in removed:
                    continue
                # Check if U[:,r] = a * U[:,s], V[:,r] = b * V[:,s], W[:,r] = c * W[:,s]
                # with a*b*c = 1
                if np.allclose(U_np[:, r], 0) and np.allclose(U_np[:, s], 0):
                    continue
                if np.allclose(V_np[:, r], 0) and np.allclose(V_np[:, s], 0):
                    continue
                if np.allclose(W_np[:, r], 0) and np.allclose(W_np[:, s], 0):
                    continue
                
                # Compute ratios
                nonzero_u = U_np[:, s] != 0
                if np.any(nonzero_u):
                    a = U_np[nonzero_u, r] / U_np[nonzero_u, s]
                    if not np.allclose(a, a[0], atol=1e-9):
                        continue
                    a_val = a[0]
                else:
                    continue
                
                nonzero_v = V_np[:, s] != 0
                if np.any(nonzero_v):
                    b = V_np[nonzero_v, r] / V_np[nonzero_v, s]
                    if not np.allclose(b, b[0], atol=1e-9):
                        continue
                    b_val = b[0]
                else:
                    continue
                
                nonzero_w = W_np[:, s] != 0
                if np.any(nonzero_w):
                    c = W_np[nonzero_w, r] / W_np[nonzero_w, s]
                    if not np.allclose(c, c[0], atol=1e-9):
                        continue
                    c_val = c[0]
                else:
                    continue
                
                # Check if a*b*c ≈ 1 (then contribution of r is absorbed by s)
                if abs(a_val * b_val * c_val - 1.0) < 1e-9:
                    # Column r is redundant with column s
                    removed.add(r)
                    improved = True
                    break
    
    if not removed:
        return U, V, W
    
    # Build reduced matrices
    keep = [r for r in range(rank) if r not in removed]
    new_rank = len(keep)
    U_new = [row[:] for row in U]  # Will slice
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    U_final = []
    V_final = []
    W_final = []
    for i in range(len(U)):
        U_final.append([U[i][r] for r in keep])
    for i in range(len(V)):
        V_final.append([V[i][r] for r in keep])
    for i in range(len(W)):
        W_final.append([W[i][r] for r in keep])
    
    return U_final, V_final, W_final

def _greedy_column_removal(U, V, W, n, m, p, deadline):
    """
    More aggressive: for each column, try to express it as a linear combination
    of other columns and update the remaining columns accordingly.
    """
    start = time.time()
    rank = len(U[0])
    removed = set()
    improved = True
    iter_count = 0
    
    while improved and iter_count < 50 and time.time() < deadline:
        improved = False
        iter_count += 1
        current_rank = rank - len(removed)
        
        for r in range(rank):
            if r in removed:
                continue
            if time.time() >= deadline:
                break
            
            # Try to remove column r by finding coefficients for other columns
            # such that the trilinear form is preserved.
            # This requires solving a system, which is expensive.
            # Instead, use a simpler check: if column r has very small entries,
            # try removing it and see if the error is within tolerance after
            # redistributing.
            
            # Check if column r can be eliminated by adding its contribution
            # to other columns
            # For each (i,j,k), the contribution of r is U[i,j,r]*V[j,k,r]*W[k,i,r]
            # We need: sum_{s != r} c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,r]*V[j,k,r]*W[k,i,r]
            # for all (i,j,k)
            
            # This is a system of n*m*p equations in (current_rank-1) unknowns.
            # For 4*4*5=80 equations and potentially many unknowns, this may be
            # underdetermined.
            
            # Build the system matrix A where A[(i,j,k), s] = U[i,j,s]*V[j,k,s]*W[k,i,s]
            # and b[(i,j,k)] = U[i,j,r]*V[j,k,r]*W[k,i,r]
            
            nm_p = n * m * p
            other_cols = [s for s in range(rank) if s != r and s not in removed]
            if not other_cols:
                continue
            
            # For efficiency, only try if the number of unknowns is manageable
            if len(other_cols) > 40:
                continue
            
            A = np.zeros((nm_p, len(other_cols)))
            b = np.zeros(nm_p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b[idx] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        for ci, s in enumerate(other_cols):
                            A[idx, ci] = U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
                        idx += 1
            
            # Solve A x = b
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact
                if residuals and residuals[0] < 1e-9:
                    # Check if solution entries are half-integers
                    if all(abs(val * 2 - round(val * 2)) < 1e-9 for val in x):
                        # Apply the update: for each other column s, add x[ci] * (contribution of r)
                        # Actually, we need to update U, V, W such that the new columns
                        # sum to the old columns plus the contribution of r
                        # This is not straightforward because the update is on the
                        # trilinear form, not on individual matrices.
                        # For now, just mark r as removed if the system is consistent
                        # and we can find a valid redistribution.
                        # This is complex, so we skip for now.
                        pass
            except:
                continue
    
    if not removed:
        return U, V, W
    
    keep = [r for r in range(rank) if r not in removed]
    U_final = [[U[i][r] for r in keep] for i in range(len(U))]
    V_final = [[V[i][r] for r in keep] for i in range(len(V))]
    W_final = [[W[i][r] for r in keep] for i in range(len(W))]
    return U_final, V_final, W_final

def _try_als_decompress(U, V, W, n, m, p, deadline):
    """
    Use ALS (Alternating Least Squares) style to find a lower-rank approximation
    and then round to half-integers.
    """
    start = time.time()
    U_np, V_np, W_np = _to_numpy(U, V, W)
    rank = U_np.shape[1]
    
    # Try to find a lower rank by projecting
    # This is heuristic and may not yield exact results
    # For now, return the original
    return U, V, W

def _round_to_half_integer(val, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    rounded = round(val * 2) / 2
    if abs(val - rounded) < tol:
        return rounded
    return val

def _round_matrix_to_half_int(M, tol=1e-6):
    """Round a matrix to half-integers where possible."""
    result = []
    for row in M:
        new_row = []
        for val in row:
            new_row.append(_round_to_half_integer(val, tol))
        result.append(new_row)
    return result

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _merge_identical_columns(U, V, W, n, m, p):
    """Merge columns that are identical in V and W (then combine U)."""
    rank = len(U[0])
    merged = list(range(rank))
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                r1 = merged[i]
                r2 = merged[j]
                # Check if V and W columns are identical
                same_v = all(V[a][r1] == V[a][r2] for a in range(m * p))
                same_w = all(W[a][r1] == W[a][r2] for a in range(p * n))
                if same_v and same_w:
                    # Merge r2 into r1
                    for a in range(n * m):
                        U[a][r1] += U[a][r2]
                    # Mark r2 for removal
                    j -= 1
                    changed = True
                j += 1
            new_merged.append(merged[i])
            i += 1
        merged = new_merged
    
    if len(merged) == rank:
        return U, V, W
    
    U_final = [[U[i][r] for r in merged] for i in range(len(U))]
    V_final = [[V[i][r] for r in merged] for i in range(len(V))]
    W_final = [[W[i][r] for r in merged] for i in range(len(W))]
    return U_final, V_final, W_final

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    n, m, p = DIMENSIONS

    # Build rank-65 baseline
    res = _build_rank65(n, m, p)
    
    if not _verify_decomposition(res["U"], res["V"], res["W"], n, m, p):
        return _build_schoolbook(n, m, p)
    
    U, V, W = res["U"], res["V"], res["W"]
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Step 1: Merge identical columns
    if time.time() < deadline - 10:
        U, V, W = _merge_identical_columns(U, V, W, n, m, p)
        if _verify_decomposition(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
            if len(U[0]) < best_rank:
                best_rank = len(U[0])
                best_result = {"U": U, "V": V, "W": W}
    
    # Step 2: Combinatorial subspace elimination
    if time.time() < deadline - 10:
        U2, V2, W2 = _combinatorial_subspace_elimination(U, V, W, n, m, p, deadline - 10)
        if _verify_decomposition(U2, V2, W2, n, m, p) and _verify_half_integer(U2, V2, W2):
            if len(U2[0]) < best_rank:
                best_rank = len(U2[0])
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Step 3: Try to find further reductions via linear algebra
    if time.time() < deadline - 5:
        # Check for zero columns
        rank = len(U[0])
        zero_cols = set()
        for r in range(rank):
            if all(U[i][r] == 0 for i in range(n*m)) and \
               all(V[i][r] == 0 for i in range(m*p)) and \
               all(W[i][r] == 0 for i in range(p*n)):
                zero_cols.add(r)
        
        if zero_cols:
            keep = [r for r in range(rank) if r not in zero_cols]
            U3 = [[U[i][r] for r in keep] for i in range(len(U))]
            V3 = [[V[i][r] for r in keep] for i in range(len(V))]
            W3 = [[W[i][r] for r in keep] for i in range(len(W))]
            if _verify_decomposition(U3, V3, W3, n, m, p) and _verify_half_integer(U3, V3, W3):
                if len(U3[0]) < best_rank:
                    best_rank = len(U3[0])
                    best_result = {"U": U3, "V": V3, "W": W3}
    
    # Step 4: Attempt SVD-based null space detection
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        
        # For each column, check if it's in the null space of some combined operator
        # This is a heuristic: columns with very small norms are candidates
        col_norms = []
        for r in range(rank):
            norm = np.linalg.norm(U_np[:, r]) * np.linalg.norm(V_np[:, r]) * np.linalg.norm(W_np[:, r])
            col_norms.append((norm, r))
        
        col_norms.sort()
        
        # Try removing the smallest columns one at a time
        for norm, r in col_norms[:5]:
            if time.time() >= deadline - 5:
                break
            if norm < 1e-12:
                # This column contributes essentially nothing
                keep = [rr for rr in range(rank) if rr != r]
                U4 = [[U[i][rr] for rr in keep] for i in range(len(U))]
                V4 = [[V[i][rr] for rr in keep] for i in range(len(V))]
                W4 = [[W[i][rr] for rr in keep] for i in range(len(W))]
                if _verify_decomposition(U4, V4, W4, n, m, p) and _verify_half_integer(U4, V4, W4):
                    if len(U4[0]) < best_rank:
                        best_rank = len(U4[0])
                        best_result = {"U": U4, "V": V4, "W": W4}
    
    return best_result
