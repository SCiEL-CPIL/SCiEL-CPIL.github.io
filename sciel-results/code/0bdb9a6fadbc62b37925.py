"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Uses iterative hard thresholding (IHT) to find sparse exact decompositions.
    The approach treats the factor matrices as a single optimization problem
    and uses sparsity constraints to minimize rank.
    """
    import numpy as np
    import itertools
    import random
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a, b, c] = 1 iff j==j' and k==k' and i==i'
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def validate(U, V, W, tol=1e-8):
        """Check if decomposition is valid."""
        T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(U.shape[1]):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_recon, T, atol=tol)
    
    def count_nonzero(U, V, W):
        """Count total non-zero entries."""
        return np.count_nonzero(U) + np.count_nonzero(V) + np.count_nonzero(W)
    
    def iht_decompose(rank, max_iters=200, tol=1e-8, seed=42):
        """
        Use IHT to find a sparse decomposition of given rank.
        Returns (U, V, W) if found, else None.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with random sparse matrices
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Random initialization with some structure
        for r in range(rank):
            # Pick random support for each factor
            u_support = rng.choice(n*m, size=min(3, n*m), replace=False)
            v_support = rng.choice(m*p, size=min(4, m*p), replace=False)
            w_support = rng.choice(p*n, size=min(4, p*n), replace=False)
            
            for idx in u_support:
                U[idx, r] = rng.choice([-1.0, 1.0])
            for idx in v_support:
                V[idx, r] = rng.choice([-1.0, 1.0])
            for idx in w_support:
                W[idx, r] = rng.choice([-1.0, 1.0])
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iters):
            # Compute current reconstruction
            T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
            for r in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                return best_U, best_V, best_W
            
            # IHT step: compute gradient and update
            # For tensor completion, use a simple iterative approach
            # Update each factor by solving a least squares problem
            
            # Update U
            for r in range(rank):
                # Compute effective target for U[:, r]
                # T = sum_r U_r *o* V_r *o* W_r
                # For fixed r, we want U_r to minimize ||T - sum_s U_s *o* V_s *o* W_s||
                # This is complex; use a simpler heuristic
                
                # Compute residual
                T_recon_other = np.zeros((n*m, m*p, p*n), dtype=float)
                for s in range(rank):
                    if s != r:
                        T_recon_other += np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
                
                residual = T - T_recon_other
                # We want U[:, r] *o* V[:, r] *o* W[:, r] ≈ residual
                # This is a rank-1 approximation problem
                
                # Use a simple approach: project residual onto the subspace spanned by V_r and W_r
                # For each (b, c), the contribution to U[a, r] is:
                # residual[a, b, c] ≈ U[a, r] * V[b, r] * W[c, r]
                # So U[a, r] ≈ sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r] / sum_{b,c} V[b,r]^2 * W[c,r]^2
                
                v_w = V[:, r] * W[:, r]  # This doesn't make sense dimensionally
                # Actually V is (m*p, rank) and W is (p*n, rank), so they have different sizes
                # Let's use a different approach
                
                # Flatten and use SVD-like approach
                # For fixed r, we have residual as a 3D tensor
                # We want to find u such that u *o* V[:, r] *o* W[:, r] ≈ residual
                # This means: for each (b, c), residual[:, b, c] ≈ u * (V[b, r] * W[c, r])
                # So u ≈ (sum_{b,c} residual[:, b, c] * V[b, r] * W[c, r]) / (sum_{b,c} V[b, r]^2 * W[c, r]^2)
                
                # Compute numerator and denominator
                num = np.zeros(n*m, dtype=float)
                den = 0.0
                for b in range(m*p):
                    for c in range(p*n):
                        vw = V[b, r] * W[c, r]
                        if abs(vw) > 1e-10:
                            num += residual[:, b, c] * vw
                            den += vw * vw
                
                if den > 1e-10:
                    U[:, r] = num / den
                else:
                    U[:, r] = 0
            
            # Hard thresholding: keep only top-k entries per column
            for r in range(rank):
                # Keep top 3 entries in U[:, r]
                u_vals = np.abs(U[:, r])
                if np.sum(u_vals > 1e-6) > 3:
                    threshold = np.sort(u_vals)[-3]
                    U[np.abs(U[:, r]) < threshold, r] = 0
                    U[:, r][np.abs(U[:, r]) < 1e-6] = 0
                    
                    # Normalize to ±1
                    for idx in np.where(np.abs(U[:, r]) > 1e-6)[0]:
                        U[idx, r] = np.sign(U[idx, r])
                
                # Keep top 4 entries in V[:, r]
                v_vals = np.abs(V[:, r])
                if np.sum(v_vals > 1e-6) > 4:
                    threshold = np.sort(v_vals)[-4]
                    V[np.abs(V[:, r]) < threshold, r] = 0
                    V[:, r][np.abs(V[:, r]) < 1e-6] = 0
                    
                    for idx in np.where(np.abs(V[:, r]) > 1e-6)[0]:
                        V[idx, r] = np.sign(V[idx, r])
                
                # Keep top 4 entries in W[:, r]
                w_vals = np.abs(W[:, r])
                if np.sum(w_vals > 1e-6) > 4:
                    threshold = np.sort(w_vals)[-4]
                    W[np.abs(W[:, r]) < threshold, r] = 0
                    W[:, r][np.abs(W[:, r]) < 1e-6] = 0
                    
                    for idx in np.where(np.abs(W[:, r]) > 1e-6)[0]:
                        W[idx, r] = np.sign(W[idx, r])
            
            # Check convergence
            if err < tol:
                return U, V, W
        
        # Check if best solution is valid
        T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', best_U[:, r], best_V[:, r], best_W[:, r])
        if np.allclose(T_recon, T, atol=tol):
            return best_U, best_V, best_W
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # 24
    upper_bound = n * m * p  # 72
    
    # First, get the schoolbook solution as fallback
    U_school = np.zeros((n*m, upper_bound), dtype=float)
    V_school = np.zeros((m*p, upper_bound), dtype=float)
    W_school = np.zeros((p*n, upper_bound), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find better decompositions using IHT
    best_rank = upper_bound
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from lower bound to upper bound - 1
    for rank in range(lower_bound, upper_bound):
        # Try multiple random seeds
        found = False
        for seed in range(5):
            result = iht_decompose(rank, max_iters=50, seed=seed)
            if result is not None:
                U, V, W = result
                if validate(U, V, W):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        if found:
            break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
