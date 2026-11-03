"""Matrix multiplication 3x5x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Precompute index mappings for fast error computation
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                valid_indices.append((a, b, c))
    
    def compute_error(U, V, W, rank):
        """Compute the full tensor error: sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2"""
        # For efficiency, compute the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can do this via einsum or manual loops
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # But for large rank this might be slow. Let's use the valid indices approach.
        
        # Actually, we need to check ALL entries, not just valid ones.
        # The full tensor has n*m * m*p * p*n = 15*30*18 = 8100 entries.
        # Let's compute it properly.
        
        # Method: For each rank component r, add U[:,r] outer V[:,r] outer W[:,r]
        # This is expensive for large rank. Let's use einsum.
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T - T_recon)**2)
        return err
    
    def compute_error_fast(U, V, W, rank):
        """Faster error computation using only valid indices plus sampling of invalid."""
        # Compute error on valid indices
        err_valid = 0.0
        for (a, b, c) in valid_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err_valid += (val - 1.0)**2
        
        # For invalid indices, we need to check they are 0.
        # This is expensive. Let's use a different approach:
        # Compute the full reconstruction and compare.
        # Actually, let's just use the full computation but with einsum which is fast.
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T - T_recon)**2)
        return err
    
    def als_decompose(rank, max_iters=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, solve min_Ua ||T[a,:,:] - Ua @ (V*W)^T||_F
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let Z_r = V[:,r] * W[:,r] (element-wise product, shape (m*p, p*n))
            # Then T[a,:,:] = sum_r U[a,r] * Z_r
            # This is a linear system: U[a,:] @ Z^T = T[a,:,:].flatten()
            # Where Z^T has shape (rank, m*p*p*n)
            # Actually, let's reshape.
            # T_a = T[a,:,:] has shape (m*p, p*n)
            # We want U_a (shape (rank,)) such that sum_r U_a[r] * Z_r ≈ T_a
            # where Z_r[b,c] = V[b,r] * W[c,r]
            
            # Form the matrix M_U of shape (m*p*p*n, rank)
            # M_U[idx, r] where idx = b*(p*n) + c, M_U[idx, r] = V[b,r] * W[c,r]
            # Then T_a_flat = T[a,:,:].flatten() has shape (m*p*p*n,)
            # Solve M_U @ U_a = T_a_flat
            
            # This is expensive. Let's use a more efficient approach.
            # For each a, we have (m*p*p*n) equations and rank unknowns.
            # m*p*p*n = 30*18 = 540. rank could be up to 90.
            # So we solve a 540xrank least squares problem for each of 15 values of a.
            
            U_new = np.zeros_like(U)
            for a in range(n*m):
                # Form M_U
                # M_U[idx, r] = V[b, r] * W[c, r] where idx = b*(p*n) + c
                # We can do this as: for each r, M_U[:, r] = (V[:, r] * W[:, r]).flatten()
                # But V[:, r] has shape (m*p,) and W[:, r] has shape (p*n,)
                # Their element-wise product doesn't directly give us what we need.
                # We need the outer product: V[b,r] * W[c,r] for all b,c.
                # This is np.outer(V[:,r], W[:,r]) which has shape (m*p, p*n)
                # Then flatten it.
                
                # More efficient: 
                # M_U = np.zeros((m*p*p*n, rank))
                # for r in range(rank):
                #     M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                # T_a_flat = T[a, :, :].flatten()
                # U_new[a, :] = np.linalg.lstsq(M_U, T_a_flat, rcond=None)[0]
                
                # Even more efficient using broadcasting:
                # V_W = V[:, :, None] * W[:, None, :]  # shape (m*p, p*n, rank)
                # M_U = V_W.transpose(2, 0, 1).reshape(rank, m*p*p*n).T  # shape (m*p*p*n, rank)
                
                V_W = np.einsum('br,cr->bcr', V, W)  # shape (m*p, p*n, rank)
                M_U = V_W.transpose(2, 0, 1).reshape(rank, m*p*p*n).T  # shape (m*p*p*n, rank)
                T_a_flat = T[a, :, :].flatten()
                U_new[a, :], residuals, rank_sv, sv = np.linalg.lstsq(M_U, T_a_flat, rcond=None)
            
            # Update V
            V_new = np.zeros_like(V)
            for b in range(m*p):
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b: T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
                # U_W = np.einsum('ar,cr->acr', U, W)  # shape (n*m, p*n, rank)
                # M_V = U_W.transpose(2, 0, 1).reshape(rank, n*m*p*n).T
                # T_b_flat = T[:, b, :].flatten()
                
                U_W = np.einsum('ar,cr->acr', U, W)  # shape (n*m, p*n, rank)
                M_V = U_W.transpose(2, 0, 1).reshape(rank, n*m*p*n).T  # shape (n*m*p*n, rank)
                T_b_flat = T[:, b, :].flatten()
                V_new[b, :], residuals, rank_sv, sv = np.linalg.lstsq(M_V, T_b_flat, rcond=None)
            
            # Update W
            W_new = np.zeros_like(W)
            for c in range(p*n):
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
                # U_V = np.einsum('ar,br->abr', U, V)  # shape (n*m, m*p, rank)
                # M_W = U_V.transpose(2, 0, 1).reshape(rank, n*m*m*p).T
                # T_c_flat = T[:, :, c].flatten()
                
                U_V = np.einsum('ar,br->abr', U, V)  # shape (n*m, m*p, rank)
                M_W = U_V.transpose(2, 0, 1).reshape(rank, n*m*m*p).T  # shape (n*m*m*p, rank)
                T_c_flat = T[:, :, c].flatten()
                W_new[c, :], residuals, rank_sv, sv = np.linalg.lstsq(M_W, T_c_flat, rcond=None)
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            err = compute_error_fast(U, V, W, rank)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, err
    
    # Try to find a low-rank decomposition
    # Start from known bounds. The schoolbook rank is 90.
    # Let's try ranks from low to high.
    
    best_rank = n*m*p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # First, verify schoolbook works
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    err_sb = compute_error_fast(U_sb, V_sb, W_sb, n*m*p)
    # print(f"Schoolbook rank {n*m*p}, error: {err_sb}")
    
    # Try lower ranks
    # We'll try a range of ranks. Given time constraints, let's be strategic.
    # Known: rank(3x3x3) = 23 (Bini)
    # For 3x5x6, the rank is likely between 23 and 90.
    # Let's try some key ranks.
    
    ranks_to_try = list(range(23, 91))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Try a few restarts for this rank
        found = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart)
            U, V, W, err = als_decompose(rank, max_iters=100, tol=1e-12)
            
            if err < 1e-8:
                # Found a valid decomposition
                # Check if we can round to integer/half-integer
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                err_round = compute_error_fast(U_round, V_round, W_round, rank)
                
                if err_round < 1e-8:
                    best_rank = rank
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    best_err = err_round
                    found = True
                    break
                else:
                    # Use the unrounded version if it's close enough
                    # But the problem requires integer/half-integer entries
                    # Let's try a local search around the rounded values
                    # For now, accept if error is small
                    if err < 1e-10:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        best_err = err
                        found = True
                        break
        
        if found:
            # Try to go lower
            continue
    
    # If no better rank found, use schoolbook
    if best_U is None:
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = n*m*p
    
    # Ensure entries are integer or half-integer
    def ensure_half_integer(arr):
        arr = np.round(arr * 2) / 2
        return arr
    
    best_U = ensure_half_integer(best_U)
    best_V = ensure_half_integer(best_V)
    best_W = ensure_half_integer(best_W)
    
    # Final verification
    final_err = compute_error_fast(best_U, best_V, best_W, best_rank)
    
    if final_err > 1e-6:
        # Fallback to schoolbook
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = n*m*p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
