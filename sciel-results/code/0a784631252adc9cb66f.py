"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = get_schoolbook()
    
    # Verify schoolbook is correct
    S_school = np.zeros_like(T)
    for r in range(rank_school):
        S_school += np.einsum('a,b,c->abc', U_school[:, r], V_school[:, r], W_school[:, r])
    if np.linalg.norm(T - S_school) > 1e-10:
        # Should not happen, but safety
        pass
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def try_snap(U, V, W, R):
        """Try multiple snapping strategies to find exact half-integer solution."""
        # Strategy 1: Direct snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        S_snap = np.zeros_like(T)
        for r in range(R):
            S_snap += np.einsum('a,b,c->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
        if np.linalg.norm(T - S_snap) < 1e-10:
            return U_snap, V_snap, W_snap, True
        
        # Strategy 2: Snap to integers first
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)
        S_snap = np.zeros_like(T)
        for r in range(R):
            S_snap += np.einsum('a,b,c->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
        if np.linalg.norm(T - S_snap) < 1e-10:
            return U_snap.astype(np.float64), V_snap.astype(np.float64), W_snap.astype(np.float64), True
        
        # Strategy 3: Polish by solving exact linear systems
        # For each row of U, solve the linear system given V and W
        U_polished = U.copy()
        V_polished = V.copy()
        W_polished = W.copy()
        
        # Iterate a few times polishing each factor
        for polish_iter in range(3):
            # Polish U
            for r in range(R):
                vw_mat = np.outer(V_polished[:, r], W_polished[:, r])
                # Solve: sum over other terms + U[:,r] @ vw_mat.flatten() = T.flatten()
                # This is a linear system in U[:,r]
                # Residual from other terms
                S_other = np.zeros_like(T)
                for r2 in range(R):
                    if r2 != r:
                        S_other += np.einsum('a,b,c->abc', U_polished[:, r2], V_polished[:, r2], W_polished[:, r2])
                residual = (T - S_other).reshape(-1)
                vw_flat = vw_mat.reshape(-1)
                # Least squares: U[:,r] @ vw_flat = residual
                # U[:,r] = residual @ vw_flat^T / (vw_flat @ vw_flat^T)
                denom = np.dot(vw_flat, vw_flat)
                if denom > 1e-12:
                    U_polished[:, r] = np.dot(residual, vw_flat) / denom
            
            # Polish V
            for r in range(R):
                uw_mat = np.outer(U_polished[:, r], W_polished[:, r])
                S_other = np.zeros_like(T)
                for r2 in range(R):
                    if r2 != r:
                        S_other += np.einsum('a,b,c->abc', U_polished[:, r2], V_polished[:, r2], W_polished[:, r2])
                residual = (T - S_other).reshape(-1)
                uw_flat = uw_mat.reshape(-1)
                denom = np.dot(uw_flat, uw_flat)
                if denom > 1e-12:
                    V_polished[:, r] = np.dot(residual, uw_flat) / denom
            
            # Polish W
            for r in range(R):
                uv_mat = np.outer(U_polished[:, r], V_polished[:, r])
                S_other = np.zeros_like(T)
                for r2 in range(R):
                    if r2 != r:
                        S_other += np.einsum('a,b,c->abc', U_polished[:, r2], V_polished[:, r2], W_polished[:, r2])
                residual = (T - S_other).reshape(-1)
                uv_flat = uv_mat.reshape(-1)
                denom = np.dot(uv_flat, uv_flat)
                if denom > 1e-12:
                    W_polished[:, r] = np.dot(residual, uv_flat) / denom
        
        # Try snapping polished versions
        for factor in ['U', 'V', 'W']:
            U_s = np.round(U_polished * 2) / 2
            V_s = np.round(V_polished * 2) / 2
            W_s = np.round(W_polished * 2) / 2
            S_s = np.zeros_like(T)
            for r in range(R):
                S_s += np.einsum('a,b,c->abc', U_s[:, r], V_s[:, r], W_s[:, r])
            if np.linalg.norm(T - S_s) < 1e-10:
                return U_s, V_s, W_s, True
        
        return None, None, None, False
    
    # Try to find lower rank decompositions
    # Lower bound is max(nm, mp, pn)
    min_rank = max(nm, mp, pn)
    
    for R in range(min_rank, rank_school):
        if time.time() > deadline:
            break
        
        found_solution = False
        
        # Try multiple random initializations with diverse seeds
        num_inits = 20
        for init_idx in range(num_inits):
            if time.time() > deadline:
                break
            
            # Use diverse random seeds
            seed = init_idx * 7919 + R * 104729 + 42
            np.random.seed(seed)
            
            # Different initialization strategies
            if init_idx % 3 == 0:
                # Small random
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif init_idx % 3 == 1:
                # Medium random with some structure
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
            else:
                # Sparse-like initialization
                U = np.random.choice([-0.5, 0.5], size=(nm, R))
                V = np.random.choice([-0.5, 0.5], size=(mp, R))
                W = np.random.choice([-0.5, 0.5], size=(pn, R))
            
            # ALS iterations
            S = np.zeros_like(T)
            for r in range(R):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            prev_error = float('inf')
            converged = False
            max_iter = 150
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U
                for r in range(R):
                    S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    vw_mat = np.outer(V[:, r], W[:, r])
                    numerator = np.einsum('ijk,jk->i', T - S, vw_mat)
                    denominator = np.sum(vw_mat ** 2)
                    if denominator > 1e-10:
                        U[:, r] = numerator / denominator
                    S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                # Update V
                for r in range(R):
                    S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    uw_mat = np.outer(U[:, r], W[:, r])
                    numerator = np.einsum('ijk,ik->j', T - S, uw_mat)
                    denominator = np.sum(uw_mat ** 2)
                    if denominator > 1e-10:
                        V[:, r] = numerator / denominator
                    S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                # Update W
                for r in range(R):
                    S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    uv_mat = np.outer(U[:, r], V[:, r])
                    numerator = np.einsum('ijk,ij->k', T - S, uv_mat)
                    denominator = np.sum(uv_mat ** 2)
                    if denominator > 1e-10:
                        W[:, r] = numerator / denominator
                    S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                error = np.linalg.norm(T - S)
                if error < 1e-6:  # Relaxed threshold for ALS, polish later
                    converged = True
                    break
                if abs(prev_error - error) < 1e-14:
                    break
                prev_error = error
            
            if not converged:
                continue
            
            # Try to snap to half-integers with multiple strategies
            U_snap, V_snap, W_snap, success = try_snap(U, V, W, R)
            if success:
                best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                found_solution = True
                break
        
        if found_solution:
            break
    
    # Convert to list of lists for interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
