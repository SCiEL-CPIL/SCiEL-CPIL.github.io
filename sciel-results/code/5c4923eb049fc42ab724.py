"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for c in range(rank_sb):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank solutions
    # The constraint: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We'll try ranks from 36 down to 30
    
    for target_rank in range(36, 29, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Use gradient-free local search: try to satisfy constraints
            # Define residual: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
            # We want all residuals to be 0
            
            # Simple iterative approach: fix two matrices, solve for third
            # This is a linear system for W given U and V
            # For each (i,k): sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is: sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
            
            # Let M[i,k,r] = sum_j U[i*m+j,r]*V[j*p+k,r]
            # Then we need: sum_r M[i,k,r] * W[k*n+i,r] = 1 for all i,k
            
            # This is a linear system: for each (i,k), a dot product with W[k*n+i,:] equals 1
            # But W has different rows for different (i,k), so it's not a single linear system
            
            # Actually, W[k*n+i, r] is independent for each (k,i), so we have n*p separate linear equations
            # Each equation: sum_r M[i,k,r] * W[k*n+i,r] = 1
            # This is underdetermined if target_rank > 1, so we can find a solution if the M vectors are nonzero
            
            # Let's try: fix U and V, compute M, then solve for W
            # For each (i,k): M[i,k,:] dot W[k*n+i,:] = 1
            # We can set W[k*n+i,:] = M[i,k,:] / (M[i,k,:] dot M[i,k,:]) if norm is nonzero
            
            converged = False
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute M[i,k,r] = sum_j U[i*m+j,r]*V[j*p+k,r]
                M = np.zeros((n, p, target_rank))
                for i in range(n):
                    for k in range(p):
                        for r in range(target_rank):
                            M[i,k,r] = np.dot(U[i*m:(i+1)*m, r], V[:, k])
                        # Actually: M[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                
                # Recompute properly
                M = np.zeros((n, p, target_rank))
                for i in range(n):
                    for k in range(p):
                        for r in range(target_rank):
                            M[i,k,r] = 0.0
                            for j in range(m):
                                M[i,k,r] += U[i*m+j, r] * V[j*p+k, r]
                
                # Solve for W: for each (i,k), need M[i,k,:] dot W[k*n+i,:] = 1
                # Set W[k*n+i,:] = M[i,k,:] / ||M[i,k,:]||^2
                valid = True
                for i in range(n):
                    for k in range(p):
                        norm_sq = np.dot(M[i,k,:], M[i,k,:])
                        if norm_sq < 1e-10:
                            valid = False
                            break
                        W[k*n+i, :] = M[i,k,:] / norm_sq
                    if not valid:
                        break
                
                if not valid:
                    break
                
                # Check residual
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = 0.0
                            for r in range(target_rank):
                                res += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_res = max(max_res, abs(res - 1.0))
                
                if max_res < 1e-6:
                    converged = True
                    break
                
                # Update U and V using similar approach
                # For U: for each (i,j), need sum_k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # Let N[i,j,r] = sum_k V[j*p+k,r]*W[k*n+i,r]
                # Then sum_r U[i*m+j,r]*N[i,j,r] = 1
                # Set U[i*m+j,:] = N[i,j,:] / ||N[i,j,:]||^2
                
                N = np.zeros((n, m, target_rank))
                for i in range(n):
                    for j in range(m):
                        for r in range(target_rank):
                            N[i,j,r] = 0.0
                            for k in range(p):
                                N[i,j,r] += V[j*p+k, r] * W[k*n+i, r]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        norm_sq = np.dot(N[i,j,:], N[i,j,:])
                        if norm_sq < 1e-10:
                            valid = False
                            break
                        U[i*m+j, :] = N[i,j,:] / norm_sq
                    if not valid:
                        break
                
                if not valid:
                    break
                
                # Similarly for V
                # For V: for each (j,k), need sum_i sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # Let P[j,k,r] = sum_i U[i*m+j,r]*W[k*n+i,r]
                # Then sum_r V[j*p+k,r]*P[j,k,r] = 1
                
                P = np.zeros((m, p, target_rank))
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            P[j,k,r] = 0.0
                            for i in range(n):
                                P[j,k,r] += U[i*m+j, r] * W[k*n+i, r]
                
                valid = True
                for j in range(m):
                    for k in range(p):
                        norm_sq = np.dot(P[j,k,:], P[j,k,:])
                        if norm_sq < 1e-10:
                            valid = False
                            break
                        V[j*p+k, :] = P[j,k,:] / norm_sq
                    if not valid:
                        break
                
                if not valid:
                    break
            
            if converged:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found a solution for this rank, try lower
    
    # Convert to lists with integer/half-integer entries if possible
    # For now, return as floats; the evaluator should accept them
    # Try to round to nearest half-integer
    def to_half_int_array(arr):
        result = np.round(arr * 2) / 2
        # Check if rounding is exact
        if np.allclose(result, arr, atol=1e-6):
            return result
        return arr
    
    U_out = to_half_int_array(best_U)
    V_out = to_half_int_array(best_V)
    W_out = to_half_int_array(best_W)
    
    # Verify the solution
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-4:
                        return False
        return True
    
    if not verify(U_out, V_out, W_out, n, m, p):
        # Fall back to schoolbook
        U_out, V_out, W_out = U_sb, V_sb, W_sb
    
    # Convert to lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

