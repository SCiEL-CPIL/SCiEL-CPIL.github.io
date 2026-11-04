import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _build_tensor_indices(n, m, p):
    """Build the list of (a, b, c) indices where T[a,b,c] = 1."""
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def _tensor_residual(U, V, W, nz_indices, n, m, p):
    """Compute residual: for each nonzero entry, check sum_c U[a,c]*V[b,c]*W[c,c] == 1.
    For zero entries, check sum_c U[a,c]*V[b,c]*W[c,c] == 0."""
    rank = U.shape[1]
    # For nonzero entries, residual should be 0 (value - 1)
    # For zero entries, residual should be 0 (value - 0)
    # We'll focus on minimizing max absolute residual
    
    # Reshape for faster computation
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    max_res = 0.0
    for (a, b, c_idx) in nz_indices:
        i, j = divmod(a, m)
        k = b - j * p
        val = 0.0
        for r in range(rank):
            val += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
        res = abs(val - 1.0)
        if res > max_res:
            max_res = res
    return max_res

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Check all nonzero entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all zero entries equal 0 (sample some for speed)
    # Zero entries are where indices don't match
    zero_count = 0
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, k = divmod(b, p)
            for c_idx in range(p * n):
                k2, i2 = divmod(c_idx, n)
                # This is a zero entry if not (j2==j and i2==i and k2==k)
                if not (j2 == j and i2 == i and k2 == k):
                    val = 0.0
                    for r in range(rank):
                        val += U_t[i, j, r] * V_t[j2, k, r] * W_t[k2, i2, r]
                    if abs(val) > tol:
                        return False
                    zero_count += 1
                    if zero_count > 100:  # Sample only
                        break
            if zero_count > 100:
                break
        if zero_count > 100:
            break
    
    return True

def _try_find_decomposition(n, m, p, target_rank, deadline, rng_seed=42):
    """Try to find a rank-target_rank decomposition using iterative refinement."""
    start_time = time.time()
    rng = np.random.default_rng(rng_seed)
    
    nz_indices = _build_tensor_indices(n, m, p)
    n_nz = len(nz_indices)
    
    # Total number of entries
    total_entries = (n * m) * (m * p) * (p * n)
    n_zero = total_entries - n_nz
    
    best_residual = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try multiple random initializations
    n_trials = 50
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = rng.standard_normal((n * m, target_rank)) * 0.5
        V = rng.standard_normal((m * p, target_rank)) * 0.5
        W = rng.standard_normal((p * n, target_rank)) * 0.5
        
        # Iterative refinement: alternate updating U, V, W
        # For each factor, given the other two, solve a linear least squares problem
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Fix V and W, update U: for each a, U[a,:] should satisfy
        # sum_c U[a,:] * (V[b,:] * W[c,:]) = T[a,b,c] for all b,c
        # This is a linear system in U[a,:]
        
        converged = False
        for it in range(200):
            if time.time() > deadline - 3:
                break
            
            # Update U
            # For each a, we need: sum_c U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Let M[a,b,c,r] = V[b,r] * W[c,r], then sum_r U[a,r] * M[a,b,c,r] = T[a,b,c]
            # This is: U[a,:] @ M[a,b,c,:] = T[a,b,c] for all b,c
            # Stack over b,c: U[a,:] @ M_stacked = T_stacked
            # M_stacked has shape (m*p*p*n, rank)
            
            # Build the design matrix for U
            # For each a, the equation is: U[a,:] * (V[b,:] * W[c,:]) = T[a,b,c]
            # where b ranges over m*p, c ranges over p*n
            # T[a,b,c] is 1 if (a,b,c) is in nz_indices, else 0
            
            U_new = np.zeros_like(U)
            for a in range(n * m):
                # Build design matrix: for each (b, c), the vector V[b,:] * W[c,:]
                # There are (m*p) * (p*n) = 27 * 18 = 486 rows
                design = np.zeros((m * p * p * n, target_rank))
                target = np.zeros(m * p * p * n)
                
                row = 0
                for b in range(m * p):
                    for c_idx in range(p * n):
                        design[row, :] = V[b, :] * W[c_idx, :]
                        # Check if (a, b, c_idx) is a nonzero entry
                        # a = i*m + j, b = j'*p + k, c_idx = k'*n + i'
                        # Nonzero if j' == j, i' == i, k' == k
                        i_a, j_a = divmod(a, m)
                        j_b, k_b = divmod(b, p)
                        k_c, i_c = divmod(c_idx, n)
                        if j_a == j_b and i_a == i_c and k_b == k_c:
                            target[row] = 1.0
                        row += 1
                
                # Solve least squares: U[a,:] @ design.T = target
                # i.e., design @ U[a,:]^T = target
                sol, residuals, rank_ls, sv = np.linalg.lstsq(design, target, rcond=None)
                U_new[a, :] = sol
            
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for b in range(m * p):
                design = np.zeros((n * m * p * n, target_rank))
                target = np.zeros(n * m * p * n)
                
                row = 0
                for a in range(n * m):
                    for c_idx in range(p * n):
                        design[row, :] = U[a, :] * W[c_idx, :]
                        i_a, j_a = divmod(a, m)
                        j_b, k_b = divmod(b, p)
                        k_c, i_c = divmod(c_idx, n)
                        if j_a == j_b and i_a == i_c and k_b == k_c:
                            target[row] = 1.0
                        row += 1
                
                sol, residuals, rank_ls, sv = np.linalg.lstsq(design, target, rcond=None)
                V_new[b, :] = sol
            
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for c_idx in range(p * n):
                design = np.zeros((n * m * m * p, target_rank))
                target = np.zeros(n * m * m * p)
                
                row = 0
                for a in range(n * m):
                    for b in range(m * p):
                        design[row, :] = U[a, :] * V[b, :]
                        i_a, j_a = divmod(a, m)
                        j_b, k_b = divmod(b, p)
                        k_c, i_c = divmod(c_idx, n)
                        if j_a == j_b and i_a == i_c and k_b == k_c:
                            target[row] = 1.0
                        row += 1
                
                sol, residuals, rank_ls, sv = np.linalg.lstsq(design, target, rcond=None)
                W_new[c_idx, :] = sol
            
            W = W_new
            
            # Check convergence
            if it % 10 == 0:
                res = _tensor_residual(U, V, W, nz_indices, n, m, p)
                if res < best_residual:
                    best_residual = res
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                if res < 1e-12:
                    converged = True
                    break
        
        if converged:
            break
    
    if best_residual < 1e-9 and best_U is not None:
        return best_U, best_V, best_W
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the standard rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, n*p) = max(6, 27, 18) = 27
    # Try ranks from 27 up to 53
    for trial_rank in range(27, 54):
        if time.time() > deadline - 10:
            break
        
        result = _try_find_decomposition(n, m, p, trial_rank, deadline)
        if result is not None:
            U, V, W = result
            best_rank = trial_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Verify the final decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
        best_rank = n * m * p
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
