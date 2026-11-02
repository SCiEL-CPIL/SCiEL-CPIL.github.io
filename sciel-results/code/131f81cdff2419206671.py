import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_trivial = n * m * p
    
    # Trivial decomposition
    U_triv = [[0] * rank_trivial for _ in range(n * m)]
    V_triv = [[0] * rank_trivial for _ in range(m * p)]
    W_triv = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    # Precompute Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W, rank):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reconstruct tensor
        # recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
        
        # Check against T
        # We need max abs diff to be very small
        if np.max(np.abs(recon - T)) > 1e-6:
            return False
        return True

    def round_half_int(M):
        """
        Attempts to round M to nearest half-integers.
        Returns (rounded_matrix, success_bool)
        """
        M2 = 2.0 * M
        Mr = np.round(M2)
        # Check if close to integers
        if np.max(np.abs(M2 - Mr)) < 1e-5:
            return Mr / 2.0, True
        return None, False

    def als_decompose(rank, max_iters=100, tol=1e-8):
        # Initialize
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # For each a in 0..nm-1:
            # Solve (V^T * W elementwise reshaped)^T @ U[a,:] = T[a,:,:].flatten()
            # M_U shape: (mp*pn, rank)
            # M_U[b*pn+c, r] = V[b,r] * W[c,r]
            MW = np.einsum('br,cr->bcr', V, W, optimize=True) # (mp, pn, rank)
            MW_flat = MW.reshape(mp*pn, rank)
            
            for a in range(n*m):
                target_a = T[a].reshape(mp*pn)
                # Solve MW_flat.T @ U[a,:] = target_a
                # MW_flat.T is (rank, mp*pn)
                # Using lstsq
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(MW_flat.T, target_a, rcond=None)
                    U[a, :] = sol
                except:
                    return None

            # Update V
            # For each b in 0..mp-1:
            # Solve (U^T * W elementwise reshaped)^T @ V[b,:] = T[:,b,:].flatten()
            # M_V shape: (nm*pn, rank)
            # M_V[a*pn+c, r] = U[a,r] * W[c,r]
            UW = np.einsum('ar,cr->acr', U, W, optimize=True) # (nm, pn, rank)
            UW_flat = UW.reshape(nm*pn, rank)
            
            for b in range(m*p):
                target_b = T[:, b, :].reshape(nm*pn)
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(UW_flat.T, target_b, rcond=None)
                    V[b, :] = sol
                except:
                    return None

            # Update W
            # For each c in 0..pn-1:
            # Solve (U^T * V elementwise reshaped)^T @ W[c,:] = T[:,:,c].flatten()
            # M_W shape: (nm*mp, rank)
            # M_W[a*mp+b, r] = U[a,r] * V[b,r]
            UV = np.einsum('ar,br->abr', U, V, optimize=True) # (nm, mp, rank)
            UV_flat = UV.reshape(nm*mp, rank)
            
            for c in range(p*n):
                target_c = T[:, :, c].reshape(nm*mp)
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(UV_flat.T, target_c, rcond=None)
                    W[c, :] = sol
                except:
                    return None

            # Check convergence
            recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.max(np.abs(recon - T))
            if err < tol:
                break
        
        # Final check and rounding
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        
        if err < 1e-6:
            # Try to round to half-integers
            U_r, ok1 = round_half_int(U)
            if ok1:
                V_r, ok2 = round_half_int(V)
                if ok2:
                    W_r, ok3 = round_half_int(W)
                    if ok3:
                        if verify(U_r, V_r, W_r, rank):
                            return U_r, V_r, W_r
        return None

    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    # Try to find lower rank
    # Lower bound is roughly max(n*m, m*p, p*n) = 30, but likely higher.
    # We search from 89 down to 70.
    for rank in range(rank_trivial - 1, 69, -1):
        if time.time() > deadline:
            break
        
        # Number of restarts decreases as we go deeper if time is tight
        n_restarts = 5 if rank > 80 else 2
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            result = als_decompose(rank)
            if result is not None:
                U_r, V_r, W_r = result
                # Convert to lists of ints/half-ints
                def to_list(mat):
                    return [[float(x) for x in row] for row in mat]
                
                return {
                    "U": to_list(U_r),
                    "V": to_list(V_r),
                    "W": to_list(W_r)
                }
            
            # Optional: If we found a good approximate solution but not exact, 
            # we might try to polish it, but for now we just move to next restart/rank.
            
    # If no better rank found, return trivial
    return best_result
