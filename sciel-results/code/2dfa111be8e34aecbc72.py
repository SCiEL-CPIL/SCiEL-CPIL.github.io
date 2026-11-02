# EVOLVE-BLOCK-START

def _compute_residual_fast(U, V, W, n, m, p, rank):
    """
    Compute residual tensor R[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    Uses vectorized operations where possible, but the triple sum is the bottleneck.
    We can rewrite the sum as a tensor contraction.
    T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    Let's reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank).
    Then T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    This is a tensor network contraction.
    We can do:
    Temp1[i, j, k, r] = U[i,j,r] * V[j,k,r]  -- Shape (n, m, p, rank)
    Then T[i, j, k] = sum_r Temp1[i, j, k, r] * W[k, i, r]
    
    This might use a lot of memory if rank is large, but here rank <= 45, so (3,3,5,45) is small.
    """
    U_res = U.reshape(n, m, rank)
    V_res = V.reshape(m, p, rank)
    W_res = W.reshape(p, n, rank)
    
    # Contract U and V over rank
    # U_res: (n, m, rank)
    # V_res: (m, p, rank)
    # We want sum_r U[i,j,r] V[j,k,r]
    # This is a batched matrix multiplication if we reshape appropriately?
    # Or just einsum: 'ijr,jkr->ijk'
    UV = np.einsum('ijr,jkr->ijk', U_res, V_res, optimize=True)
    
    # Now contract with W
    # UV: (n, m, p)
    # W_res: (p, n, rank)
    # We want sum_r UV[i,j,k] W[k,i,r] ... wait, the sum is over r.
    # The previous step summed over r? No, einsum 'ijr,jkr->ijk' sums over r.
    # So UV[i,j,k] = sum_r U[i,j,r] V[j,k,r].
    # But the formula is sum_r (U[i,j,r] V[j,k,r] W[k,i,r]).
    # The W term also depends on r. So we cannot sum U and V first.
    
    # We need: T[i,j,k] = sum_r (U[i,j,r] * V[j,k,r] * W[k,i,r])
    
    # Let's form a tensor A[i,j,k,r] = U[i,j,r] * V[j,k,r]
    # Shape: (n, m, p, rank)
    # Then T[i,j,k] = sum_r A[i,j,k,r] * W[k,i,r]
    
    # Memory check: 3*3*5*45 = 2025 floats. Very small.
    A = np.einsum('ijr,jkr->ijk r', U_res, V_res, optimize=True)
    # Note: 'ijk r' is just 'ijk r'
    
    # Now contract A and W over r
    # A: (n, m, p, rank)
    # W_res: (p, n, rank)
    # We want sum_r A[i,j,k,r] * W[k,i,r]
    # Result shape: (n, m, p)
    T = np.einsum('ijk r,k i r->ijk', A, W_res, optimize=True)
    
    R = 1.0 - T
    return R

def _local_search_rank_r(n, m, p, rank, rng, deadline, initial=None):
    """
    Local search for a rank-r decomposition minimizing sum of squared residuals.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    if initial is not None:
        U, V, W = initial.astype(np.float64)
    else:
        # Initialize with small random integers
        U = rng.integers(-1, 2, size=(N, rank)).astype(np.float64)
        V = rng.integers(-1, 2, size=(P, rank)).astype(np.float64)
        W = rng.integers(-1, 2, size=(Q, rank)).astype(np.float64)

    max_iters = 50000
    # We will update one variable at a time to minimize the global error.
    # Error E = ||R||^2 = sum_{i,j,k} (1 - sum_r U[i,j,r]V[j,k,r]W[k,i,r])^2
    
    # To speed up, we can precompute some terms?
    # Let's just do a simple coordinate descent.
    # Pick a random variable x in {U, V, W} at a random index.
    # Compute the gradient or optimal value for x given the others.
    
    # Optimal value for U[i,j,r]:
    # E = sum_{k} (1 - U[i,j,r]V[j,k,r]W[k,i,r] - Rest_{i,j,k,r})^2
    # Let A_{i,j,k} = 1 - Rest_{i,j,k,r} where Rest excludes the term with U[i,j,r]
    # Actually, Rest depends on r.
    # Let S_{i,j,k} = sum_{r' != r} U[i,j,r']V[j,k,r']W[k,i,r']
    # Term for r: U[i,j,r] V[j,k,r] W[k,i,r]
    # Residual R_{i,j,k} = 1 - S_{i,j,k} - U[i,j,r] V[j,k,r] W[k,i,r]
    # We want to minimize sum_k R_{i,j,k}^2.
    # This is a least squares problem for U[i,j,r].
    # min_x sum_k (1 - S_{i,j,k} - x V[j,k,r] W[k,i,r])^2
    # Let y_k = 1 - S_{i,j,k}
    # Let c_k = V[j,k,r] W[k,i,r]
    # min_x sum_k (y_k - x c_k)^2
    # x* = (sum_k y_k c_k) / (sum_k c_k^2)
    
    # Similarly for V and W.
    
    for _iter in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Choose which matrix to update
        mat_choice = rng.integers(0, 3)
        
        if mat_choice == 0:
            i = rng.integers(0, n)
            j = rng.integers(0, m)
            r = rng.integers(0, rank)
            
            # Compute y_k and c_k for k in 0..p-1
            # S_{i,j,k} = sum_{r' != r} U[i*m+j, r'] V[j*p+k, r'] W[k*n+i, r']
            # We can compute full sum T_{i,j,k} and subtract the current term.
            # But computing full T is expensive.
            # Let's just compute the sum for this specific i,j.
            
            # Vectorize over k
            # U_val = U[i*m+j, r]
            # V_vec = V[j*p : (j+1)*p, r]  # shape (p,)
            # W_vec = W[0::n, r]            # shape (p,) -- indices k*n+i for k=0..p-1
            # W_indices = np.arange(p) * n + i
            # W_vec = W[W_indices, r]
            
            # We need S_{i,j,k} for all k.
            # S_{i,j,k} = sum_{r' in 0..rank-1} U[i*m+j, r'] V[j*p+k, r'] W[k*n+i, r']
            #           - U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # Let's compute the full sum for this i,j using matrix multiplication.
            # U_row = U[i*m+j, :]  # shape (rank,)
            # V_block = V[j*p:(j+1)*p, :] # shape (p, rank)
            # W_block = W[0::n, :] # shape (p, rank) -- wait, W index is k*n+i.
            # W_indices = np.arange(p) * n + i
            # W_block = W[W_indices, :] # shape (p, rank)
            
            # T_k = sum_r U_row[r] V_block[k, r] W_block[k, r]
            # This is element-wise product of V_block and W_block, then dot with U_row.
            # VW = V_block * W_block # shape (p, rank)
            # T_vec = VW @ U_row # shape (p,)
            
            U_row = U[i * m + j, :]
            V_block = V[j * p:(j + 1) * p, :]
            W_indices = np.arange(p) * n + i
            W_block = W[W_indices, :]
            
            VW = V_block * W_block
            T_vec = VW @ U_row
            
            # Now subtract the term for r
            term_r = U_row[r] * V_block[:, r] * W_block[:, r]
            S_vec = T_vec - term_r
            
            y_vec = 1.0 - S_vec
            c_vec = V_block[:, r] * W_block[:, r]
            
            denom = np.sum(c_vec * c_vec)
            if denom < 1e-12:
                continue
                
            new_val = np.sum(y_vec * c_vec) / denom
            
            # Round to nearest integer or half-integer?
            # The problem allows integer/half-integer.
            # Let's try rounding to nearest 0.5.
            new_val_rounded = np.round(new_val * 2) / 2.0
            
            # Check if this improves the error
            # Old error contribution: sum_k (y_vec - U_row[r] * c_vec)^2
            # New error contribution: sum_k (y_vec - new_val_rounded * c_vec)^2
            
            old_err = np.sum((y_vec - U_row[r] * c_vec) ** 2)
            new_err = np.sum((y_vec - new_val_rounded * c_vec) ** 2)
            
            if new_err < old_err - 1e-12:
                U[i * m + j, r] = new_val_rounded
            else:
                # Try integer rounding
                new_val_int = np.round(new_val)
                new_err_int = np.sum((y_vec - new_val_int * c_vec) ** 2)
                if new_err_int < old_err - 1e-12:
                    U[i * m + j, r] = new_val_int

        elif mat_choice == 1:
            j = rng.integers(0, m)
            k = rng.integers(0, p)
            r = rng.integers(0, rank)
            
            # Update V[j*p+k, r]
            # Symmetric logic.
            # Variable is V_val.
            # y_i = 1 - S_{i,j,k}
            # c_i = U[i*m+j, r] W[k*n+i, r]
            
            # Compute S_{i,j,k} for all i.
            # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # S_{i,j,k} = T_{i,j,k} - U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # V_val = V[j*p+k, r]
            # U_indices = np.arange(n) * m + j
            # U_block = U[U_indices, :] # shape (n, rank)
            # W_indices = k * n + np.arange(n)
            # W_block = W[W_indices, :] # shape (n, rank)
            
            # T_i = sum_r U_block[i, r] V_val W_block[i, r]
            # UW = U_block * W_block
            # T_vec = UW @ [0..0, 1 at r, 0..0]? No.
            # T_i = V_val * sum_r U_block[i, r] W_block[i, r]
            # SumVec = np.sum(U_block * W_block, axis=1) # shape (n,)
            # T_vec = V_val * SumVec
            
            # S_vec = T_vec - U_block[:, r] * V_val * W_block[:, r]
            # y_vec = 1 - S_vec
            # c_vec = U_block[:, r] * W_block[:, r]
            
            V_val = V[j * p + k, r]
            U_indices = np.arange(n) * m + j
            U_block = U[U_indices, :]
            W_indices = k * n + np.arange(n)
            W_block = W[W_indices, :]
            
            UW = U_block * W_block
            SumVec = np.sum(UW, axis=1)
            T_vec = V_val * SumVec
            
            term_r = U_block[:, r] * V_val * W_block[:, r]
            S_vec = T_vec - term_r
            
            y_vec = 1.0 - S_vec
            c_vec = U_block[:, r] * W_block[:, r]
            
            denom = np.sum(c_vec * c_vec)
            if denom < 1e-12:
                continue
                
            new_val = np.sum(y_vec * c_vec) / denom
            new_val_rounded = np.round(new_val * 2) / 2.0
            
            old_err = np.sum((y_vec - V_val * c_vec) ** 2)
            new_err = np.sum((y_vec - new_val_rounded * c_vec) ** 2)
            
            if new_err < old_err - 1e-12:
                V[j * p + k, r] = new_val_rounded
            else:
                new_val_int = np.round(new_val)
                new_err_int = np.sum((y_vec - new_val_int * c_vec) ** 2)
                if new_err_int < old_err - 1e-12:
                    V[j * p + k, r] = new_val_int

        else:
            k = rng.integers(0, p)
            i = rng.integers(0, n)
            r = rng.integers(0, rank)
            
            # Update W[k*n+i, r]
            # Variable is W_val.
            # y_j = 1 - S_{i,j,k}
            # c_j = U[i*m+j, r] V[j*p+k, r]
            
            # Compute S_{i,j,k} for all j.
            # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # S_{i,j,k} = T_{i,j,k} - U[i*m+j, r] V[j*p+k, r] W_val
            
            W_val = W[k * n + i, r]
            U_indices = np.arange(m) + i * m # i*m + j for j=0..m-1
            U_block = U[U_indices, :]
            V_indices = k + np.arange(m) * p # j*p + k for j=0..m-1
            V_block = V[V_indices, :]
            
            UV = U_block * V_block
            SumVec = np.sum(UV, axis=1)
            T_vec = W_val * SumVec
            
            term_r = U_block[:, r] * V_block[:, r] * W_val
            S_vec = T_vec - term_r
            
            y_vec = 1.0 - S_vec
            c_vec = U_block[:, r] * V_block[:, r]
            
            denom = np.sum(c_vec * c_vec)
            if denom < 1e-12:
                continue
                
            new_val = np.sum(y_vec * c_vec) / denom
            new_val_rounded = np.round(new_val * 2) / 2.0
            
            old_err = np.sum((y_vec - W_val * c_vec) ** 2)
            new_err = np.sum((y_vec - new_val_rounded * c_vec) ** 2)
            
            if new_err < old_err - 1e-12:
                W[k * n + i, r] = new_val_rounded
            else:
                new_val_int = np.round(new_val)
                new_err_int = np.sum((y_vec - new_val_int * c_vec) ** 2)
                if new_err_int < old_err - 1e-12:
                    W[k * n + i, r] = new_val_int

    # Final check
    R = _compute_residual_fast(U, V, W, n, m, p, rank)
    if np.max(np.abs(R)) < 1e-9:
        # Convert to list of ints/floats
        # Ensure they are exact integers or halves
        U_out = np.round(U * 2) / 2.0
        V_out = np.round(V * 2) / 2.0
        W_out = np.round(W * 2) / 2.0
        
        # Verify again with rounded values
        R2 = _compute_residual_fast(U_out, V_out, W_out, n, m, p, rank)
        if np.max(np.abs(R2)) < 1e-9:
            return (U_out, V_out, W_out)
            
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0  # Safety margin

    rng = np.random.default_rng(42)

    ranks_to_try = [41, 42, 43, 44]
    
    for rank in ranks_to_try:
        if time.time() >= deadline:
            break
        
        time_per_rank = 25.0 if rank == 41 else 15.0
        rank_deadline = min(deadline, time.time() + time_per_rank)
        
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() >= rank_deadline:
                break
                
            sol = _local_search_rank_r(n, m, p, rank, rng, rank_deadline)
            if sol is not None:
                U, V, W = sol
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist(),
                }

    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
