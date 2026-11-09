"""Matrix multiplication 3,3,6 - rank-minimal tensor decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p = 54
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=np.float64)
    V_school = np.zeros((m*p, rank_school), dtype=np.float64)
    W_school = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Strategy: exploit structure B = [B1, B2] where each Bi is 3x3
    # AB = [A*B1, A*B2]
    # If we have a rank-r decomposition for 3x3*3x3, we get rank 2r for 3x3*3x6
    # Known: rank(3x3 mult) is between 19 and 23 (best known is 19 by Bläser)
    # But we can use Strassen-like methods to get rank 7 for 2x2, and extend.
    # 
    # For 3x3*3x3, we can use a rank 19 decomposition if we can find one.
    # However, let's first try to find good decompositions via structured search.
    
    # Let's try to use the structure more cleverly.
    # A is 3x3, B is 3x6 = [B1 | B2] where B1, B2 are 3x3.
    # AB = A*B1 | A*B2
    # 
    # We can try to find a decomposition of the 3x3*3x6 tensor directly.
    # The tensor has size 9*18*18 = 2916.
    # 
    # Let's use a random search with gradient descent on the factor matrices.
    # We'll minimize ||sum_c U_c ⊗ V_c ⊗ W_c - T||_F^2
    
    rng = np.random.RandomState(42)
    
    def compute_residual(U, V, W):
        """Compute the residual tensor sum_c U_c ⊗ V_c ⊗ W_c - T"""
        rank = U.shape[1]
        residual = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(rank):
            residual += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        return residual - T
    
    def compute_error(U, V, W):
        """Compute squared Frobenius norm of residual"""
        residual = compute_residual(U, V, W)
        return np.sum(residual**2)
    
    # Target ranks to try
    # Since 3x3*3x3 has rank at most 19, 3x3*3x6 should have rank at most 38.
    # But maybe we can do better with a direct approach.
    # Let's try ranks from 20 to 54.
    
    target_ranks = list(range(20, 55))
    
    found_better = False
    for target_r in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_attempts = 3 if target_r <= 30 else 2
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random values
            scale = 0.5
            U = rng.randn(n*m, target_r) * scale
            V = rng.randn(m*p, target_r) * scale
            W = rng.randn(p*n, target_r) * scale
            
            current_error = compute_error(U, V, W)
            
            if current_error < 1e-10:
                best_rank = target_r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                found_better = True
                break
            
            # Simple iterative improvement: coordinate descent style
            # For each factor and each term, try to minimize error
            max_iterations = 50
            for iteration in range(max_iterations):
                if time.time() > deadline - 20:
                    break
                
                improved = False
                
                # Update U
                for c in range(target_r):
                    # Fix V_c and W_c, optimize U_c
                    # residual = T - sum_{c' != c} U_c' ⊗ V_c' ⊗ W_c'
                    # We want to minimize ||U_c ⊗ V_c ⊗ W_c - residual||_F^2
                    # This is equivalent to finding U_c that best approximates
                    # the "contraction" of residual with V_c and W_c
                    
                    # Compute residual without term c
                    residual_without_c = compute_residual(
                        U, V, W
                    ) + np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                    residual_without_c = T - residual_without_c
                    
                    # Optimal U_c = argmin ||U_c ⊗ V_c ⊗ W_c - residual_without_c||_F^2
                    # This is a linear system. The optimal U_c is:
                    # U_c[a] = sum_{b,d} residual_without_c[a,b,d] * V_c[b] * W_c[d] / (sum_b V_c[b]^2 * sum_d W_c[d]^2)
                    
                    v_norm_sq = np.sum(V[:,c]**2)
                    w_norm_sq = np.sum(W[:,c]**2)
                    
                    if v_norm_sq < 1e-12 or w_norm_sq < 1e-12:
                        continue
                    
                    new_U_c = np.array([
                        np.sum(residual_without_c[a,:,:] * V[:,c] * W[:,c]) / (v_norm_sq * w_norm_sq)
                        for a in range(n*m)
                    ])
                    
                    old_error = current_error
                    U[:,c] = new_U_c
                    new_error = compute_error(U, V, W)
                    
                    if new_error < old_error:
                        current_error = new_error
                        improved = True
                    else:
                        # Revert
                        pass  # U already updated, but error didn't improve, so we accept it for now
        
        if found:
            break
    
    # If we couldn't find a better decomposition, return the schoolbook
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
