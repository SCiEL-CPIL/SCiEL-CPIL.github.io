"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # seconds, safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition
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
    
    # Build the target tensor T[i,j,k] = 1 if output index matches
    # T is a 3D array: T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a, b, c] = sum over rank-1 terms: U[a,k] * V[b,k] * W[c,k]
    # For schoolbook: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to reduce rank by checking for linear dependencies
    # Use a greedy approach: try removing each column and check if tensor is still matched
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if U, V, W exactly decompose the matmul tensor."""
        r = U_mat.shape[1]
        # Compute the reconstructed tensor
        # T_recon[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
        # Compare with target tensor
        # Target: T_target[i*m+j, j*p+k, k*n+i] = 1, else 0
        
        # Efficient check: for each (a,b) pair, compute vector v[c] = sum_k U[a,k]*V[b,k]*W[c,k]
        # and check it matches target
        
        # Build target tensor as sparse
        # Target entries: for i in range(n), j in range(m), k in range(p):
        #   a = i*m+j, b = j*p+k, c = k*n+i, value = 1
        
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Compute sum_k U[a,k]*V[b,k]*W[c,k]
                    val = np.dot(U_mat[a, :] * V_mat[b, :], W_mat[c, :])
                    if abs(val - 1) > tol:
                        return False
                    # Also check that non-target entries are 0
                    # This is expensive, so we'll do a random sample
        
        # Random check of non-target entries
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(n * m)
            b = rng.randint(m * p)
            c = rng.randint(p * n)
            # Check if this is a target entry
            is_target = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i * m + j == a and j * p + k == b and k * n + i == c:
                            is_target = True
                            break
                    if is_target:
                        break
                if is_target:
                    break
            
            val = np.dot(U_mat[a, :] * V_mat[b, :], W_mat[c, :])
            if is_target:
                if abs(val - 1) > tol:
                    return False
            else:
                if abs(val) > tol:
                    return False
        
        return True
    
    def try_remove_column(U_mat, V_mat, W_mat, col_idx):
        """Try to absorb column col_idx into other columns."""
        r = U_mat.shape[1]
        if r <= 1:
            return None, None, None
        
        # Try to express column col_idx as linear combination of others
        # This is a complex problem; instead, try simple cases
        
        # Check if U[:, col_idx] is a multiple of some other column
        # and similarly for V and W
        
        u_col = U_mat[:, col_idx].copy()
        v_col = V_mat[:, col_idx].copy()
        w_col = W_mat[:, col_idx].copy()
        
        # Try each other column as potential "absorber"
        for other in range(r):
            if other == col_idx:
                continue
            
            u_other = U_mat[:, other].copy()
            v_other = V_mat[:, other].copy()
            w_other = W_mat[:, other].copy()
            
            # Check if u_col is scalar multiple of u_other
            if np.allclose(u_other, 0):
                continue
            
            # Find alpha such that u_col = alpha * u_other
            nonzero_mask = np.abs(u_other) > 1e-12
            if np.sum(nonzero_mask) == 0:
                continue
            
            alpha = np.dot(u_col[nonzero_mask], u_other[nonzero_mask]) / np.dot(u_other[nonzero_mask], u_other[nonzero_mask])
            
            if np.allclose(u_col, alpha * u_other, atol=1e-10):
                # Check if v_col is also alpha * v_other
                if np.allclose(v_col, alpha * v_other, atol=1e-10):
                    # Then we need w_col contribution to be absorbed
                    # New w_other = w_other + (1/alpha) * w_col * (something)
                    # Actually this gets complex. Let's just check if we can remove.
                    pass
        
        return None, None, None
    
    # Simpler structural reduction: exploit that for fixed (i,j), varying k gives p terms
    # that share the same U column. Similarly for other structures.
    
    # Alternative approach: use known low-rank decompositions
    # For 3x3 x 3x3, the minimal rank is 23 (Landsberg et al.)
    # For 3x3 x 3x8, we can try to use a block approach
    
    # Let's try a randomized local search to reduce rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try greedy removal: for each column, check if removing it still gives exact decomposition
    # after adjusting other columns (this is hard, so we'll use a simpler heuristic)
    
    # Structural insight: The tensor has a specific sparsity pattern.
    # Let's try to find a lower-rank decomposition using the fact that
    # the problem has symmetries.
    
    # For now, return schoolbook if we can't find better
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a simple rank reduction: merge columns that have identical U and V
    # patterns (up to scaling)
    
    merged_U = []
    merged_V = []
    merged_W = []
    used = [False] * rank
    
    for i in range(rank):
        if used[i]:
            continue
        used[i] = True
        u_i = U[:, i].copy()
        v_i = V[:, i].copy()
        w_i = W[:, i].copy()
        
        # Check if any later column can be merged
        for j in range(i + 1, rank):
            if used[j]:
                continue
            u_j = U[:, j].copy()
            v_j = V[:, j].copy()
            w_j = W[:, j].copy()
            
            # Check if (u_j, v_j) is proportional to (u_i, v_i)
            # Find alpha such that u_j = alpha * u_i
            if np.allclose(u_i, 0):
                continue
            
            nonzero = np.abs(u_i) > 1e-12
            if np.sum(nonzero) == 0:
                continue
            
            alpha = np.dot(u_j[nonzero], u_i[nonzero]) / np.dot(u_i[nonzero], u_i[nonzero])
            
            if np.allclose(u_j, alpha * u_i, atol=1e-10) and np.allclose(v_j, alpha * v_i, atol=1e-10):
                # Can merge: new_w = w_i + w_j (since the alpha factors cancel in the sum)
                # Wait, need to be more careful
                # Term i: u_i ⊗ v_i ⊗ w_i
                # Term j: u_j ⊗ v_j ⊗ w_j = alpha * u_i ⊗ alpha * v_i ⊗ w_j = alpha^2 * u_i ⊗ v_i ⊗ w_j
                # Sum: u_i ⊗ v_i ⊗ (w_i + alpha^2 * w_j)
                # So we can merge by updating w_i
                w_i = w_i + (alpha * alpha) * w_j
                used[j] = True
        
        merged_U.append(u_i)
        merged_V.append(v_i)
        merged_W.append(w_i)
    
    if len(merged_U) < rank:
        new_U = np.column_stack(merged_U)
        new_V = np.column_stack(merged_V)
        new_W = np.column_stack(merged_W)
        
        if check_decomposition(new_U, new_V, new_W):
            best_U, best_V, best_W = new_U, new_V, new_W
            best_rank = len(merged_U)
    
    if time.time() > deadline - 3:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
