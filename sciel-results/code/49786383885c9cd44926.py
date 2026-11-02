"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 110.0 # Safety margin for 120s limit

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Search for lower ranks
    # Start from 59 down to 50. 
    # If we find a solution at r, we continue to search for r-1.
    for r in range(59, 49, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W with random small integers
            # Using randint to encourage integer solutions
            U = np.random.randint(-2, 3, (n*m, r)).astype(float)
            V = np.random.randint(-2, 3, (m*p, r)).astype(float)
            W = np.random.randint(-2, 3, (p*n, r)).astype(float)
            
            # Local search using alternating minimization / power method
            converged = False
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                    
                # Compute residual
                current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
                residual = T - current_sum
                
                # Check if residual is small enough
                if np.linalg.norm(residual) < 1e-9:
                    converged = True
                    break
                
                # Update one column at a time? Or all?
                # Let's update all columns simultaneously using a gradient step or power method on the tensor
                # Power method for tensor decomposition is tricky. 
                # Let's use a simple alternating least squares approach on the flattened matrices?
                # Or just perturb?
                
                # A better approach: For each column c, try to optimize U[:, c], V[:, c], W[:, c]
                # to minimize || T - sum_{c'} U_{:, c'} V_{:, c'} W_{:, c'} ||^2
                # This is hard to solve exactly for rank 1.
                
                # Let's stick to the power method idea from inspiration 2 but applied to the whole tensor?
                # Or just random restarts with no local search? 
                # Given the small dimensions, maybe we can just try random integer vectors?
                # Rank 59 is very close to 60. The solution is likely to be very close to schoolbook.
                
                # Let's try a different strategy: 
                # Start with schoolbook rank 60. 
                # Try to set one column to zero and adjust others?
                # This is complex.
                
                # Let's use the local search from inspiration 2 more carefully.
                # It updates one column c at a time.
                c = np.random.randint(r)
                
                # Compute residual excluding column c
                sum_without_c = current_sum - np.einsum('ic,jc,kc->ijk', U[:, c:c+1], V[:, c:c+1], W[:, c:c+1])
                target_for_c = T - sum_without_c
                
                # Find best rank-1 approx to target_for_c
                # Use power method
                u = U[:, c].copy()
                v = V[:, c].copy()
                w = W[:, c].copy()
                
                # If vectors are zero, reinitialize
                if np.linalg.norm(u) < 1e-6: u = np.random.randn(n*m)
                if np.linalg.norm(v) < 1e-6: v = np.random.randn(m*p)
                if np.linalg.norm(w) < 1e-6: w = np.random.randn(p*n)
                
                for _ in range(10): # Few power iterations
                    # u_new = target_for_c contracted with v, w
                    u_new = np.einsum('ijk,j,k->i', target_for_c, v, w)
                    v_new = np.einsum('ijk,i,k->j', target_for_c, u, w)
                    w_new = np.einsum('ijk,i,j->k', target_for_c, u, v)
                    
                    norm_u = np.linalg.norm(u_new)
                    norm_v = np.linalg.norm(v_new)
                    norm_w = np.linalg.norm(w_new)
                    
                    if norm_u == 0 or norm_v == 0 or norm_w == 0:
                        break
                    
                    u_new /= norm_u
                    v_new /= norm_v
                    w_new /= norm_w
                    
                    u, v, w = u_new, v_new, w_new
                
                # Scale to match magnitude
                uo = np.einsum('i,j,k->ijk', u, v, w)
                denom = np.sum(uo * uo)
                if denom == 0:
                    continue
                s = np.sum(target_for_c * uo) / denom
                
                U[:, c] = s * u
                V[:, c] = s * v
                W[:, c] = s * w
                
            if converged:
                # We have a real-valued solution.
                # Try to round to half-integers.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    # Convert to lists
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    # Found a valid rank r. 
                    # We can break the restart loop and try lower rank.
                    break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
