# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin for 120s limit
    
    # First, get the trivial decomposition as fallback
    def trivial_decomposition():
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
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = trivial_decomposition()
    
    def build_tensor():
        """Build the full matrix multiplication tensor."""
        # T has shape (n*m, m*p, p*n)
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j      # row of A
                    b = j * p + k      # col of B  
                    c = k * n + i      # row of C (trace convention W[k*n+i])
                    T[a, b, c] = 1
        return T
    
    T = build_tensor()
    
    def check_decomposition_fast(U, V, W, rank, T):
        """Faster check using tensor operations."""
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is the mode-wise product
        # For each r, we have an outer product U[:,r] outer V[:,r] outer W[:,r]
        # Sum over r
        # Use einsum
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(reconstructed, T, atol=1e-10):
            return False
        return True
        
    def rational_reconstruct(arr, max_denom=20):
        """
        Attempt to reconstruct each entry of arr as a simple fraction p/q.
        Returns a new array with rational values.
        """
        res = np.zeros_like(arr, dtype=float)
        for idx in np.ndindex(arr.shape):
            val = arr[idx]
            # Check if close to integer
            if abs(val - round(val)) < 1e-6:
                res[idx] = round(val)
                continue
            # Try small denominators
            found = False
            for q in range(2, max_denom + 1):
                p = round(val * q)
                if abs(val - p / q) < 1e-6:
                    res[idx] = p / q
                    found = True
                    break
            if not found:
                # Keep original if no simple fraction found
                res[idx] = val
        return res

    def try_random_als(rank, max_iters=200, tol=1e-9):
        """Try to find a rank-`rank` decomposition using random ALS with SVD init."""
        # SVD Initialization
        # Mode-0 unfolding of T: T_0 has shape (n*m, m*p * p*n)
        T_0 = T.reshape(n * m, -1)
        U, s, Vt = np.linalg.svd(T_0, full_matrices=False)
        
        # Project to rank
        if rank < len(s):
            U_init = U[:, :rank] * s[:rank]
        else:
            U_init = U * s
        
        # This gives U. We need V and W.
        # We can initialize V and W randomly or via other mode SVDs.
        # Let's initialize V and W randomly for simplicity, or use mode-1 SVD for V.
        
        T_1 = np.transpose(T, (1, 0, 2)).reshape(m * p, -1)
        V_svd, s1, Wt_svd = np.linalg.svd(T_1, full_matrices=False)
        if rank < len(s1):
            V_init = V_svd[:, :rank] * s1[:rank]
        else:
            V_init = V_svd * s1
            
        T_2 = np.transpose(T, (2, 0, 1)).reshape(p * n, -1)
        W_svd, s2, Ut_svd = np.linalg.svd(T_2, full_matrices=False)
        if rank < len(s2):
            W_init = W_svd[:, :rank] * s2[:rank]
        else:
            W_init = W_svd * s2

        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Normalize to prevent scale issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                U[:, r] /= norm_u
                V[:, r] /= norm_v
                # W will be updated in ALS to fit.

        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            VW = np.einsum('br,cr->bcr', V, W)
            for a in range(n * m):
                M_a = VW[:, :, :].reshape(m * p, p * n, rank).transpose(2, 0, 1).reshape(rank, m * p * p * n)
                rhs = T[a, :, :].reshape(-1)
                try:
                    U[a, :] = np.linalg.lstsq(M_a, rhs, rcond=None)[0]
                except:
                    U[a, :] = np.random.randn(rank)
            
            # Update V
            UW = np.einsum('ar,cr->acr', U, W)
            for b in range(m * p):
                M_b = UW[:, :, :].reshape(n * m, p * n, rank).transpose(2, 0, 1).reshape(rank, n * m * p * n)
                rhs = T[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(M_b, rhs, rcond=None)[0]
                except:
                    V[b, :] = np.random.randn(rank)
            
            # Update W
            UV = np.einsum('ar,br->abr', U, V)
            for c in range(p * n):
                M_c = UV[:, :, :].reshape(n * m, m * p, rank).transpose(2, 0, 1).reshape(rank, n * m * m * p)
                rhs = T[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(M_c, rhs, rcond=None)[0]
                except:
                    W[c, :] = np.random.randn(rank)
            
            # Check convergence
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(reconstructed - T) / max(1, np.linalg.norm(T))
            if err < tol:
                break
        
        # Try rational reconstruction
        U_r = rational_reconstruct(U, max_denom=20)
        V_r = rational_reconstruct(V, max_denom=20)
        W_r = rational_reconstruct(W, max_denom=20)
        
        if check_decomposition_fast(U_r, V_r, W_r, rank, T):
            return U_r, V_r, W_r
            
        # Also check original floating point (in case rational reconstruction failed but FP was exact enough for allclose)
        if check_decomposition_fast(U, V, W, rank, T):
            # If FP is exact, try to snap to integers if close
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if check_decomposition_fast(U_i, V_i, W_i, rank, T):
                return U_i, V_i, W_i
            # If not integer, return FP? The task asks for integer/half-integer.
            # If FP is not close to rational, it's likely not a valid exact decomposition in the required format.
            # So we should return None unless we find a rational/integer solution.
            pass
            
        return None
    
    # Try various ranks, starting from lower
    for rank in range(23, 64):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for each rank
        max_restarts = 20 if rank < 40 else 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            result = try_random_als(rank, max_iters=100)
            if result is not None:
                U, V, W = result
                # Verify
                if check_decomposition_fast(U, V, W, rank, T):
                    # Found a valid decomposition
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    break
        
        if best_rank == rank:
            break
    
    # Convert to lists for the return interface
    def to_list(arr):
        return [[float(val) for val in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
