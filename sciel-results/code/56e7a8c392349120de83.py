import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _build_base_decomposition(n, m, p):
    """Build the base rank-36 decomposition using Strassen 2x2x2 blocks + schoolbook."""
    strassen_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],
        [0,1,0,-1],
        [0,0,1,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
    ]
    
    rank = 4 * 7 + 8  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li * m + j0 + lj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0 + lj) * p + k0 + lk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # Schoolbook for k=4: 8 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return U, V, W, rank

def _compute_residual(U, V, W, n, m, p, cols_to_remove):
    """Compute the residual tensor after removing specified columns.
    Returns a 3D array R[i,j,k] = sum over remaining columns of U*V*W - target.
    """
    rank = len(U[0])
    R = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    if c not in cols_to_remove:
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                R[i,j,k] = total - 1.0
    return R

def _try_merge_pair(U, V, W, n, m, p, a, b, tol=1e-9):
    """Try to merge columns a and b into a single column.
    Returns (new_U, new_V, new_W) if successful, else None.
    The merged column replaces column a, and column b is removed.
    """
    rank = len(U[0])
    # Compute residual from removing both a and b
    remove_set = {a, b}
    R = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    if c not in remove_set:
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                R[i,j,k] = total - 1.0
    
    # Try to find rank-1 decomposition of R: R[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    # First, check if R is rank-1 by examining its matrix unfolding
    # Flatten to (n*m, p) and check rank
    R_mat = R.reshape(n*m, p)
    
    # If R is all zeros, we can just remove both columns (but that's handled separately)
    if np.max(np.abs(R_mat)) < tol:
        return None  # Both can be removed, handled by single removal
    
    # Try to find rank-1 approximation
    # Use SVD to check if rank-1
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(R_mat, full_matrices=False)
    except:
        return None
    
    # Check if rank-1 (second singular value ~ 0)
    if len(s_vals) > 1 and s_vals[1] > 1e-6:
        return None  # Not rank-1
    
    # R is rank-1: R = s[0] * u[:,0] * Vt[0,:]
    # We need R[i,j,k] = u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
    # This is more complex. Let's use a different approach.
    
    # Alternative: parameterize the merged column
    # New column c has values U_c[i,j], V_c[j,k], W_c[k,i]
    # We need: sum_{c' != a,b} U[c']V[c']W[c'] + U_c*V_c*W_c = 1 for all (i,j,k)
    # i.e., U_c*V_c*W_c = R (the residual)
    
    # For rank-1 R, we can set:
    # Pick a reference (i0,j0,k0) where R[i0,j0,k0] != 0
    # Then U_c[i,j] = R[i,j,k0] / (V_c[j,k0] * W_c[k0,i])
    # This gets circular. Let's use a simpler approach:
    
    # Set U_c[i,j] = R[i,j,0] for all i,j (if R[i,j,0] works)
    # Then V_c[j,k] = R[j,k,0] / (U_c[j,0] * W_c[0,j]) -- still circular
    
    # Better: use the fact that if R is rank-1 as a tensor, we can factor it.
    # R[i,j,k] = A[i,j] * B[j,k] * C[k,i]
    
    # Try: set C[k,i] = 1 for all k,i (absorb into A and B)
    # Then R[i,j,k] = A[i,j] * B[j,k]
    # Check: R_mat = A_col * B_row where A_col = A.reshape(n*m) and B_row = B.reshape(m*p)
    # Actually R[i,j,k] = A[i,j] * B[j,k] means R_mat[i,j,k] = A[i,j] * B[j,k]
    # This is a rank-1 matrix in the (n*m, p) unfolding only if A and B are consistent.
    
    # Let's just try: A[i,j] = R[i,j,0] (using k=0)
    # B[j,k] = R[j,k,0] / A[j,0] if A[j,0] != 0
    A = R[:, :, 0].copy()  # A[i,j]
    B = np.zeros((m, p), dtype=float)
    valid = False
    for j in range(m):
        for k in range(p):
            if abs(A[0, j]) > tol:
                B[j, k] = R[0, j, k] / A[0, j]
                valid = True
            elif abs(A[1, j]) > tol:
                B[j, k] = R[1, j, k] / A[1, j]
                valid = True
            else:
                B[j, k] = 0
    
    if not valid:
        return None
    
    # Verify: R[i,j,k] ≈ A[i,j] * B[j,k]
    R_approx = np.einsum('ij,jk->ijk', A, B)
    if np.max(np.abs(R - R_approx)) > 1e-6:
        # Try another factorization
        # Use SVD-based approach
        # R[i,j,k] = sum_r sigma_r u_r[i,j] v_r[j,k] w_r[k,i]
        # For rank-1, just take the first term
        # This is getting complex. Let's use a numerical optimization approach.
        return None
    
    # Now construct the new column
    # U_c[i,j] = A[i,j], V_c[j,k] = B[j,k], W_c[k,i] = 1
    new_rank = rank - 1
    new_U = [row[:] for row in U]
    new_V = [row[:] for row in V]
    new_W = [row[:] for row in W]
    
    # Remove column b (shift columns after b left by 1)
    for mat in [new_U, new_V, new_W]:
        for i in range(len(mat)):
            for c in range(b, rank-1):
                mat[i][c] = mat[i][c+1]
            mat[i][rank-1] = 0
    
    # Set merged column at position a
    for i in range(n):
        for j in range(m):
            new_U[i*m+j][a] = A[i,j]
    for j in range(m):
        for k in range(p):
            new_V[j*p+k][a] = B[j,k]
    for k in range(p):
        for i in range(n):
            new_W[k*n+i][a] = 1.0
    
    return new_U, new_V, new_W

def _try_rank_reduction(U, V, W, n, m, p, time_limit=60):
    """
    Attempt to reduce the rank by merging columns.
    """
    start_time = time.time()
    rank = len(U[0])
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = rank
    
    def _verify_fast(U_mat, V_mat, W_mat, keep_indices):
        """Fast verification."""
        if not keep_indices:
            return True
        U_k = U_mat[:, keep_indices]
        V_k = V_mat[:, keep_indices]
        W_k = W_mat[:, keep_indices]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_k[i*m+j] * V_k[j*p+k] * W_k[k*n+i])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_keep = list(range(rank))
    
    # Phase 1: Try single column removals
    changed = True
    while changed and (time.time() - start_time) < time_limit * 0.3:
        changed = False
        for c in list(current_keep):
            if (time.time() - start_time) > time_limit * 0.3:
                break
            new_keep = [i for i in current_keep if i != c]
            if _verify_fast(U_arr, V_arr, W_arr, new_keep):
                current_keep = new_keep
                best_rank = len(new_keep)
                changed = True
    
    # Phase 2: Try pair merges
    if (time.time() - start_time) < time_limit * 0.8:
        changed = True
        while changed and (time.time() - start_time) < time_limit * 0.8:
            changed = False
            for a_idx in range(len(current_keep)):
                if (time.time() - start_time) > time_limit * 0.8:
                    break
                for b_idx in range(a_idx + 1, len(current_keep)):
                    if (time.time() - start_time) > time_limit * 0.8:
                        break
                    a = current_keep[a_idx]
                    b = current_keep[b_idx]
                    
                    result = _try_merge_pair(U, V, W, n, m, p, a, b)
                    if result is not None:
                        new_U, new_V, new_W = result
                        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                            U, V, W = new_U, new_V, new_W
                            U_arr = np.array(U, dtype=float)
                            V_arr = np.array(V, dtype=float)
                            W_arr = np.array(W, dtype=float)
                            current_keep = list(range(len(U[0])))
                            best_rank = len(current_keep)
                            best_U = [row[:] for row in U]
                            best_V = [row[:] for row in V]
                            best_W = [row[:] for row in W]
                            changed = True
                            break
                if changed:
                    break
    
    if best_rank < rank:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Round to nearest 0.5
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-9:
                        mat[i][j] = rounded
        
        return U_list, V_list, W_list, best_rank
    
    return None

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Build the base rank-36 solution
    U, V, W, rank = _build_base_decomposition(n, m, p)
    
    # Verify the base solution
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    time_remaining = 140 - (time.time() - start_time)
    if time_remaining > 15:
        result = _try_rank_reduction(U, V, W, n, m, p, time_limit=min(time_remaining - 5, 60))
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < rank:
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    U, V, W, rank = U_new, V_new, W_new, new_rank
    
    return {"U": U, "V": V, "W": W}
