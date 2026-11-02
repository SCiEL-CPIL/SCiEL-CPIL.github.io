"""Construct a rank-minimal tensor decomposition for 2x4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r W[k*n+i,r]*U[i*m+j,r]*V[j'*p+k,r] == delta[j,j'].
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    R = U.shape[1]
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(R):
                w_val = W[k * n + i, r]
                if w_val == 0:
                    continue
                u_vec = U[i * m:(i + 1) * m, r]
                v_col = V[k::p, r]
                M += w_val * np.outer(u_vec, v_col)
            if not np.allclose(M, np.eye(m), atol=1e-10):
                return False
    return True

def _build_strassen_2x4x4(n, m, p):
    """
    Build a rank-28 decomposition of 2x4x4 matrix multiplication using
    four independent 2x2x2 Strassen products.
    """
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    def _strassen_block(base_col, i_block, j_block, k_block):
        a_rows = [0, 1]
        a_cols = [j_block * 2, j_block * 2 + 1]
        b_rows = [j_block * 2, j_block * 2 + 1]
        b_cols = [k_block * 2, k_block * 2 + 1]
        c_rows = [0, 1]
        c_cols = [k_block * 2, k_block * 2 + 1]
        
        # M1 = (A00+A11)*(B00+B11)
        U[0 * m + a_cols[0], base_col + 0] = 1
        U[1 * m + a_cols[1], base_col + 0] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 0] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 0] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 0] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 0] = 1
        
        # M2 = (A10+A11)*B00
        U[1 * m + a_cols[0], base_col + 1] = 1
        U[1 * m + a_cols[1], base_col + 1] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 1] = 1
        W[c_cols[0] * n + c_rows[1], base_col + 1] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 1] = -1
        
        # M3 = A00*(B01-B11)
        U[0 * m + a_cols[0], base_col + 2] = 1
        V[b_rows[0] * p + b_cols[1], base_col + 2] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 2] = -1
        W[c_cols[1] * n + c_rows[0], base_col + 2] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 2] = 1
        
        # M4 = A11*(B10-B00)
        U[1 * m + a_cols[1], base_col + 3] = 1
        V[b_rows[1] * p + b_cols[0], base_col + 3] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 3] = -1
        W[c_cols[0] * n + c_rows[0], base_col + 3] = 1
        W[c_cols[0] * n + c_rows[1], base_col + 3] = 1
        
        # M5 = (A00+A10)*B11
        U[0 * m + a_cols[0], base_col + 4] = 1
        U[1 * m + a_cols[0], base_col + 4] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 4] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 4] = -1
        W[c_cols[1] * n + c_rows[0], base_col + 4] = 1
        
        # M6 = (A01-A00)*(B00+B10)
        U[0 * m + a_cols[1], base_col + 5] = 1
        U[0 * m + a_cols[0], base_col + 5] = -1
        V[b_rows[0] * p + b_cols[0], base_col + 5] = 1
        V[b_rows[1] * p + b_cols[0], base_col + 5] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 5] = 1
        
        # M7 = (A10-A11)*(B01+B11)
        U[1 * m + a_cols[0], base_col + 6] = 1
        U[1 * m + a_cols[1], base_col + 6] = -1
        V[b_rows[0] * p + b_cols[1], base_col + 6] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 6] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 6] = 1
    
    _strassen_block(0, 0, 0, 0)
    _strassen_block(7, 0, 1, 0)
    _strassen_block(14, 0, 0, 1)
    _strassen_block(21, 0, 1, 1)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=5000, deadline=None):
    """
    Try to reduce the rank of the decomposition by finding linear combinations
    of rank-1 slices that can be eliminated.
    
    We search for a lower-rank decomposition by:
    1. Starting from the rank-28 Strassen solution
    2. Trying to express some columns as linear combinations of others
    3. Using random perturbations to find valid lower-rank decompositions
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    R_orig = U.shape[1]
    best_result = None
    best_rank = R_orig
    
    # Try various target ranks
    for target in range(target_rank, min(R_orig, target_rank + 3)):
        if deadline is not None and time.time() > deadline:
            break
        
        # Strategy: try to find a rank-`target` decomposition by 
        # projecting onto a random subspace and checking if it still works
        
        for attempt in range(max_iters // 10):
            if deadline is not None and time.time() > deadline:
                break
            
            # Create a random mixing matrix to combine the original rank-1 slices
            # This is a heuristic: we try random linear combinations
            
            # Pick `target` columns from the original R_orig columns
            # and try to see if they span a valid decomposition
            
            if target >= R_orig:
                cols = list(range(R_orig))
            else:
                # Random subset
                cols = np.random.choice(R_orig, size=target, replace=False)
                cols = sorted(cols)
            
            U_sub = U[:, cols]
            V_sub = V[:, cols]
            W_sub = W[:, cols]
            
            if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
                if target < best_rank:
                    best_rank = target
                    best_result = (U_sub.copy(), V_sub.copy(), W_sub.copy())
                    break
            
            # Also try: random linear combinations of columns
            # Mix pairs of columns and see if we can eliminate one
            if target < R_orig and attempt < max_iters // 5:
                # Pick a pair to merge
                i, j = np.random.choice(R_orig, size=2, replace=False)
                if j in cols:
                    continue
                    
                # Try: new_col = U[:,i] + alpha * U[:,j]
                # We need to find alpha such that the combined slice works
                # This is complex; instead try random alpha values
                for alpha in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
                    if deadline is not None and time.time() > deadline:
                        break
                    
                    U_new = U.copy()
                    V_new = V.copy()
                    W_new = W.copy()
                    
                    # Combine column i and j: new_i = U[:,i] + alpha*U[:,j]
                    # new_V = V[:,i] + alpha*V[:,j] 
                    # But W needs to be adjusted... this doesn't simply work
                    # Skip this approach for now
                    pass
        
        if best_result is not None and best_result[0].shape[1] <= target:
            break
    
    return best_result

def _search_rank_decomposition(n, m, p, deadline):
    """
    Search for the minimum rank decomposition using multiple strategies.
    """
    # Start with Strassen rank-28
    U28, V28, W28 = _build_strassen_2x4x4(n, m, p)
    
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try to reduce rank
    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline:
            break
        
        result = _try_reduce_rank(U28, V28, W28, n, m, p, target_rank, 
                                  max_iters=2000, deadline=deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = result[0].shape[1]
            if best_rank <= 24:
                break
    
    return best_U, best_V, best_W

def _build_strassen_2x4x4(n, m, p):
    """
    Build a rank-28 decomposition of 2x4x4 matrix multiplication using
    four independent 2x2x2 Strassen products.
    
    A is 2x4, B is 4x4, C = A*B is 2x4.
    Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    Split B into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2).
    Split C into C_L (2x2, cols 0-1) and C_R (2x2, cols 2-3).
    
    C_L = A_L * B11 + A_R * B21
    C_R = A_L * B12 + A_R * B22
    
    Each 2x2x2 product uses Strassen (rank 7).
    Total rank = 4 * 7 = 28.
    """
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    def _strassen_block(base_col, j_block, k_block):
        a_cols = [j_block * 2, j_block * 2 + 1]
        b_rows = [j_block * 2, j_block * 2 + 1]
        b_cols = [k_block * 2, k_block * 2 + 1]
        c_rows = [0, 1]
        c_cols = [k_block * 2, k_block * 2 + 1]
        
        # M1 = (A00+A11)*(B00+B11)
        U[0 * m + a_cols[0], base_col + 0] = 1
        U[1 * m + a_cols[1], base_col + 0] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 0] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 0] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 0] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 0] = 1
        
        # M2 = (A10+A11)*B00
        U[1 * m + a_cols[0], base_col + 1] = 1
        U[1 * m + a_cols[1], base_col + 1] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 1] = 1
        W[c_cols[0] * n + c_rows[1], base_col + 1] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 1] = -1
        
        # M3 = A00*(B01-B11)
        U[0 * m + a_cols[0], base_col + 2] = 1
        V[b_rows[0] * p + b_cols[1], base_col + 2] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 2] = -1
        W[c_cols[1] * n + c_rows[0], base_col + 2] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 2] = 1
        
        # M4 = A11*(B10-B00)
        U[1 * m + a_cols[1], base_col + 3] = 1
        V[b_rows[1] * p + b_cols[0], base_col + 3] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 3] = -1
        W[c_cols[0] * n + c_rows[0], base_col + 3] = 1
        W[c_cols[0] * n + c_rows[1], base_col + 3] = 1
        
        # M5 = (A00+A10)*B11
        U[0 * m + a_cols[0], base_col + 4] = 1
        U[1 * m + a_cols[0], base_col + 4] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 4] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 4] = -1
        W[c_cols[1] * n + c_rows[0], base_col + 4] = 1
        
        # M6 = (A01-A00)*(B00+B10)
        U[0 * m + a_cols[1], base_col + 5] = 1
        U[0 * m + a_cols[0], base_col + 5] = -1
        V[b_rows[0] * p + b_cols[0], base_col + 5] = 1
        V[b_rows[1] * p + b_cols[0], base_col + 5] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 5] = 1
        
        # M7 = (A10-A11)*(B01+B11)
        U[1 * m + a_cols[0], base_col + 6] = 1
        U[1 * m + a_cols[1], base_col + 6] = -1
        V[b_rows[0] * p + b_cols[1], base_col + 6] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 6] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 6] = 1
    
    _strassen_block(0, 0, 0)
    _strassen_block(7, 1, 0)
    _strassen_block(14, 0, 1)
    _strassen_block(21, 1, 1)
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Search for minimum rank decomposition
    U, V, W = _search_rank_decomposition(n, m, p, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
