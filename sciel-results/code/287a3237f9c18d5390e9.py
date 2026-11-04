import numpy as np
import time

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker-Strassen construction then systematic column elimination.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    U_base, V_base, W_base = _construct_rank98(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = _construct_naive(n, m, p)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    best_result = (U_base, V_base, W_base)
    best_rank = U_base.shape[1]
    
    # Try column elimination to reduce rank
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 15:
        reduced = _column_elimination_v2(U_base, V_base, W_base, n, m, p, deadline - start_time)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                if U_r.shape[1] < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = U_r.shape[1]
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}


def _construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return U, V, W


def _column_elimination_v2(U, V, W, n, m, p, time_budget):
    """
    Systematic column elimination using tensor structure analysis.
    More efficient than v1 by precomputing tensor contributions and
    using a greedy importance-based removal strategy.
    """
    start = time.time()
    rank = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    num_eqs = n * m * p  # 128 equations
    
    # Precompute tensor contribution vectors for each column
    # For column t, the contribution to tensor at (i,j,k) is U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    # We'll build a matrix where each column t represents the tensor vector for that column
    
    # Build the full tensor matrix: shape (num_eqs, rank)
    # T[i,j,k,t] = U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((num_eqs, rank), dtype=float)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(rank):
                    T[eq_idx, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                eq_idx += 1
    
    # The target tensor is all ones (since we're decomposing the identity multiplication tensor)
    target = np.ones(num_eqs, dtype=float)
    
    # Verify: sum of all columns should equal target
    col_sum = T.sum(axis=1)
    if not np.allclose(col_sum, target, atol=1e-9):
        return None
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    current_T = T.copy()
    
    # Greedy elimination: repeatedly find a column that can be expressed as
    # a linear combination of the others, and remove it
    max_iterations = current_rank
    for iteration in range(max_iterations):
        if time.time() - start > time_budget:
            break
        if current_rank <= 1:
            break
        
        # For each column, check if it can be eliminated
        # A column r can be eliminated if there exist coefficients alpha_s (s != r)
        # such that T[:, r] = sum_{s != r} alpha_s * T[:, s]
        # AND the corresponding U, V, W columns satisfy the same linear relation
        
        best_col_to_remove = -1
        best_solution = None
        best_residual = float('inf')
        
        # Try each column for elimination
        for r in range(current_rank):
            if time.time() - start > time_budget:
                break
            
            # Build the system: T[:, r] = T[:, remaining] @ alpha
            remaining = [s for s in range(current_rank) if s != r]
            if len(remaining) == 0:
                continue
            
            T_rem = current_T[:, remaining]
            T_target = current_T[:, r]
            
            # Solve least squares
            try:
                alpha, residuals, rank_A, sv = np.linalg.lstsq(T_rem, T_target, rcond=1e-10)
            except np.linalg.LinAlgError:
                continue
            
            # Check residual
            if residuals.size > 0:
                res_val = residuals[0]
            else:
                res_val = np.linalg.norm(T_rem @ alpha - T_target)
            
            if res_val > 1e-8:
                continue
            
            # Verify the solution is valid
            pred = T_rem @ alpha
            if not np.allclose(pred, T_target, atol=1e-8):
                continue
            
            # Check if the U, V, W columns also satisfy this relation
            # We need: U[:, r] = sum alpha_s * U[:, s], etc.
            # But this is NOT required! The tensor equation only requires the
            # product U*V*W to satisfy the relation, not the individual factors.
            # So we can just remove column r and scale the others by alpha.
            
            # Actually, we need to be careful. The decomposition requires:
            # sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 for all (i,j,k)
            # If we remove column r and scale column s by alpha_s, the new sum is:
            # sum_{s != r} alpha_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
            # This equals T_target[i,j,k] only if the tensor relation holds.
            # But we also need the individual U, V, W to be consistent.
            
            # The correct approach: we can't just scale columns independently.
            # We need to find new U', V', W' such that the tensor equation holds.
            # One way: keep all columns except r, and adjust them.
            
            # Actually, the simplest valid approach: if T[:, r] is in the span of
            # the other T columns, we can express the tensor contribution of column r
            # as a combination of others. But we need to update U, V, W consistently.
            
            # A valid method: for each (i,j,k), the equation is linear in the products.
            # We can absorb column r into the others by modifying U, V, or W.
            # The cleanest way: since the tensor is a sum of rank-1 terms, and we've
            # shown T_r is in the span of others, we can just drop column r and
            # verify the remaining decomposition still works. But this only works if
            # the remaining columns' tensor sum equals the target.
            
            # Let's just check: does the sum of remaining columns equal the target?
            remaining_sum = current_T[:, remaining].sum(axis=1)
            if np.allclose(remaining_sum, target, atol=1e-8):
                # This means column r contributes zero to the net tensor, so we can drop it
                # without changing anything else
                best_col_to_remove = r
                best_solution = None  # No scaling needed
                best_residual = 0.0
                break
            else:
                # Column r is not redundant in the simple sense. We need a more complex
                # elimination that modifies the remaining columns.
                # For now, skip this column.
                continue
        
        if best_col_to_remove >= 0:
            # Remove the column
            keep = [s for s in range(current_rank) if s != best_col_to_remove]
            current_U = current_U[:, keep]
            current_V = current_V[:, keep]
            current_W = current_W[:, keep]
            current_T = current_T[:, keep]
            current_rank -= 1
            continue
        
        # No simple column found to remove. Try a more aggressive approach:
        # Look for pairs of columns that can be combined.
        # For now, break if we can't find any simple removal.
        break
    
    if current_rank < rank:
        return current_U, current_V, current_W
    
    return None


def _construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        M = np.zeros(rank)
        for t in range(rank):
            u_val = np.dot(U[:, t], A_flat)
            v_val = np.dot(V[:, t], B_flat)
            M[t] = u_val * v_val
        
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = np.dot(W[k*n+i, :], M)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True
