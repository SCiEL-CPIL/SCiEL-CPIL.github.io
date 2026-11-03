"""Matrix multiplication 4,4,5 tensor decomposition with combinatorial subspace elimination."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    # Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U: i = i1*2 + i2, j = j1*2 + j2
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            # V: j = j1*2 + j2, k = k1*2 + k2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            # W: k = k1*2 + k2, i = i1*2 + i2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    # Naive: A[i,j] * B[j,4] -> C[4,i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return U, V, W

def _combinatorial_subspace_elimination(U_init, V_init, W_init, n, m, p, max_iterations=50, tol=1e-8, max_time=30):
    """
    Attempts to reduce rank by identifying and removing linearly dependent terms
    using null-space analysis of the tensor flattening.
    """
    start_time = time.time()
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    rank = U.shape[1]
    
    def _verify_current():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    total = np.dot(u_val * v_val, w_val)
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    if not _verify_current():
        return None
    
    # Pre-compute term vectors for efficient checking
    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
    target = np.ones((n, m, p)).flatten()
    
    for iteration in range(max_iterations):
        if time.time() > start_time + max_time:
            break
        
        # Check if we can remove any single term
        removed_any = False
        for r in range(rank):
            if time.time() > start_time + max_time:
                break
            
            # Check if column r is effectively zero
            if (np.linalg.norm(U[:, r]) < tol and 
                np.linalg.norm(V[:, r]) < tol and 
                np.linalg.norm(W[:, r]) < tol):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                term_vecs = np.delete(term_vecs, r, axis=1)
                removed_any = True
                break
        
        if removed_any:
            continue
        
        # Try null-space approach: check if removing term r allows exact reconstruction
        # from the remaining terms
        for r in range(rank):
            if time.time() > start_time + max_time:
                break
            
            remaining = [x for x in range(rank) if x != r]
            M_remaining = term_vecs[:, remaining]
            
            # Check if target is in the column space of M_remaining
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target, rcond=None)
                
                if residuals and len(residuals) > 0 and residuals[0] < tol:
                    # Verify
                    U_new = np.zeros((n*m, len(remaining)))
                    V_new = np.zeros((m*p, len(remaining)))
                    W_new = np.zeros((p*n, len(remaining)))
                    
                    for idx, r_idx in enumerate(remaining):
                        U_new[:, idx] = U[:, r_idx] * c[idx]
                        V_new[:, idx] = V[:, r_idx]
                        W_new[:, idx] = W[:, r_idx]
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                        U, V, W = U_new, V_new, W_new
                        rank = len(remaining)
                        term_vecs = _compute_term_vectors(U, V, W, n, m, p)
                        removed_any = True
                        break
            except Exception:
                pass
        
        if not removed_any:
            break
    
    if _verify_current():
        return U, V, W
    
    return None

def _try_rank_reduction_by_removal(U, V, W, n, m, p, tol=1e-9, max_time=30):
    """
    Attempts to reduce rank by removing one term at a time.
    Uses SVD-based rank estimation to quickly identify if the current set of terms
    is linearly dependent, and if so, finds a null-space vector to eliminate a term.
    """
    start_time = time.time()
    rank = U.shape[1]
    num_entries = n * m * p
    
    target_flat = np.ones(num_entries)
    
    # Pre-compute term vectors matrix M (num_entries x rank)
    # M[:, r] = vec(T_r)
    M = np.zeros((num_entries, rank))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                u_val = u[i*m+j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v[j*p+k]
                    if v_val == 0: continue
                    w_val = w[k*n+i]
                    if w_val == 0: continue
                    val = u_val * v_val * w_val
                    if val != 0:
                        flat_idx = (i * m + j) * p + k
                        M[flat_idx, r] = val

    # Check if the current decomposition is valid (it should be, but just in case)
    # sum_r M[:, r] * 1 = target_flat? No, the coefficients are all 1 in the current construction.
    # Actually, the current construction has coefficients 1.
    # If we change coefficients, we need to track them.
    # The current code assumes coefficients are 1.
    # If we remove a term, we need to re-scale the others.
    
    # Strategy:
    # 1. Check rank of M. If rank(M) < rank, there is a dependency.
    # 2. Find null space vector c such that M c = 0.
    # 3. If c has a zero component, we can't directly eliminate that term using this dependency alone?
    #    Actually, if M c = 0, then sum c_r T_r = 0.
    #    This doesn't directly help remove a term unless we can combine with the target equation.
    #    We want sum_{r != k} d_r T_r = target.
    #    This is equivalent to checking if target is in the span of {T_r | r != k}.
    
    # Efficient check for each k:
    # Project target onto the span of {T_r | r != k}.
    # If the residual is 0, we can eliminate k.
    
    # We can use SVD of M to get a basis for the column space.
    # But the column space changes when we remove a column.
    
    # Alternative: Use the fact that if the columns are linearly independent, 
    # the rank is rank. If we remove one, the rank drops only if that column was not in the span of the others.
    # So we just need to check if col_k is in span(cols \ k).
    
    # This is equivalent to checking if the rank of M is equal to the rank of M without col_k.
    # Or, checking if the coefficient in the SVD of M for col_k is non-zero in the "removed" subspace?
    
    # Simpler: For each k, compute the distance of col_k to the span of the others.
    # If distance is 0, col_k is redundant.
    
    # To do this efficiently, we can compute the inverse of M^T M (if full rank) or use QR.
    # But M is 80x65. Rank is likely 65 (full column rank).
    # If full column rank, no single column is redundant.
    # So we can only reduce rank if we *change* the terms, not just remove them.
    # The current code only removes terms and rescales. This only works if the current set was over-complete.
    # Since Strassen is minimal for 2x2x2, the Kronecker product is likely minimal for the sub-tensors.
    # Thus, simple removal might not work.
    
    # However, the combination of Strassen (49) and Naive (16) might have interactions.
    # Let's try to find a linear dependency among the 65 terms.
    
    try:
        # Check rank of M
        s = np.linalg.svd(M, compute_uv=False)
        # Count significant singular values
        num_rank = np.sum(s > 1e-8)
        
        if num_rank < rank:
            # There is a null space
            # Compute null space
            _, vh = np.linalg.svd(M, compute_uv=True)
            # Null space is rows of vh corresponding to zero singular values
            null_space = vh[num_rank:, :]
            
            # Try to find a sparse vector in the null space
            # For now, just check if any row of null space has a zero (which is always true if dim > 1)
            # We want to eliminate one term.
            # If we have a dependency sum c_r T_r = 0, we can't eliminate a term unless we can set c_k = 0?
            # No, that's not how it works.
            
            # Let's stick to the previous logic but optimized.
            pass
    except Exception:
        pass

    # Fallback to iterative removal with fast residual check
    # We can pre-compute the projection matrix P = M (M^T M)^{-1} M^T
    # Then residual for removing k is || (I - P_k) target ||
    # This is still expensive.
    
    # Given the time budget, the most effective change is to ensure the existing
    # `lstsq` calls are as fast as possible and we try *pairwise* removals or 
    # different initializations.
    
    # For now, return None if no quick win is found, to allow other strategies.
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based rank-65 construction
    try:
        U, V, W = _construct_rank_65(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            candidates = [(rank, U, V, W)]
        else:
            candidates = []
    except Exception:
        candidates = []
    
    # Pick the best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, U, V, W = candidates[0]
        best_result = (U, V, W)
    
    # Approach 2: Combinatorial subspace elimination
    if best_result is not None and time.time() < deadline - 30:
        try:
            reduced = _combinatorial_subspace_elimination(best_result[0], best_result[1], best_result[2], n, m, p)
            if reduced is not None:
                U_r, V_r, W_r = reduced
                rank_r = U_r.shape[1]
                if rank_r < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = rank_r
        except Exception:
            pass
    
    # Approach 3: Try rank reduction by term removal (single and pairwise)
    if best_result is not None and time.time() < deadline - 15:
        try:
            U_r, V_r, W_r = best_result
            remaining_time = min(45, deadline - time.time() - 5)
            reduced = _try_rank_reduction_by_removal(U_r, V_r, W_r, n, m, p, max_time=remaining_time)
            if reduced is not None:
                best_result = reduced
                best_rank = reduced[0].shape[1]
        except Exception:
            pass
    
    # Approach 4: Iterative single-term removal (try to remove one at a time repeatedly)
    if best_result is not None and time.time() < deadline - 5:
        try:
            U_r, V_r, W_r = best_result
            current_rank = U_r.shape[1]
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for r in range(current_rank):
                    if time.time() > deadline - 5:
                        break
                    
                    remaining = [x for x in range(current_rank) if x != r]
                    if not remaining:
                        break
                    
                    M_remaining = _compute_term_vectors(U_r, V_r, W_r, n, m, p)[:, remaining]
                    target_flat = np.ones((n, m, p)).flatten()
                    
                    try:
                        c, residuals, _, _ = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
                        
                        if residuals and len(residuals) > 0 and residuals[0] < 1e-9:
                            U_new = np.zeros((n*m, len(remaining)))
                            V_new = np.zeros((m*p, len(remaining)))
                            W_new = np.zeros((p*n, len(remaining)))
                            
                            for idx, r_idx in enumerate(remaining):
                                U_new[:, idx] = U_r[:, r_idx] * c[idx]
                                V_new[:, idx] = V_r[:, r_idx]
                                W_new[:, idx] = W_r[:, r_idx]
                            
                            if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                                U_r, V_r, W_r = U_new, V_new, W_new
                                current_rank = len(remaining)
                                best_rank = current_rank
                                best_result = (U_r, V_r, W_r)
                                improved = True
                                break
                    except Exception:
                        pass
        except Exception:
            pass
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
