import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook decomposition as guaranteed fallback
    rank_sb = n * m * p  # 84
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    def try_rank(r, num_trials=50):
        """
        Attempts to find a valid decomposition of rank r.
        Uses random V, solves for U and W via least squares.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() - start_time > DEADLINE - 10.0:
            return None

        # Pre-compute indices for the tensor T[i,j,k] = 1
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        # Strategy: 
        # 1. Generate random V (m*p, r)
        # 2. For each (i,j), U[i,j,:] is determined by W[:,i,:] and V[j,:,:]
        #    Specifically, the equation is:
        #    sum_k W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1
        #    This is not linear in U and W simultaneously.
        
        # Alternative Linear System Approach:
        # The tensor T_{ijk} can be written as sum_r U_{ijr} V_{jkr} W_{kir}.
        # Fix V. Then for each k, we have a system relating U and W.
        # Actually, it's easier to view this as:
        # For each r, let u_r be the vector of U[:,r] (size n*m), v_r be V[:,r] (size m*p), w_r be W[:,r] (size p*n).
        # This is still nonlinear.
        
        # Let's use the standard "randomized SVD" or "linear system" trick for matrix multiplication rank.
        # T_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
        # We can fix V to be random. Then the equation becomes:
        # sum_r W_{kir} U_{ijr} V_{jkr} = 1
        # This is linear in U if W is fixed, and linear in W if U is fixed.
        # We can alternate: 
        # 1. Init random V, random U, random W.
        # 2. Solve for U given V, W (linear system).
        # 3. Solve for W given U, V (linear system).
        # 4. Check residual.
        
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 5.0:
                return None
                
            # Initialize V randomly
            V = np.random.randn(m * p, r)
            # Normalize to avoid scaling issues
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            
            # Initialize U and W randomly
            U = np.random.randn(n * m, r)
            W = np.random.randn(p * n, r)
            
            converged = False
            for it in range(50):
                if time.time() - start_time > DEADLINE - 5.0:
                    return None
                
                # Solve for U:
                # For each (i,j), sum_r U[i,j,r] * (sum_k V[j,k,r] * W[k,i,r]) = 1
                # Let B[i,j,r] = sum_k V[j,k,r] * W[k,i,r]
                # Then sum_r U[i,j,r] * B[i,j,r] = 1
                # This is a system of n*m equations for U (which has n*m*r unknowns).
                # Wait, U is (n*m, r). The equation is dot(U[i,j,:], B[i,j,:]) = 1.
                # This is underdetermined if r > 1. We want a specific solution.
                # We can solve this by setting U as the least squares solution to:
                # M_U * U_flat = 1, where M_U is constructed such that M_U * U_flat = sum_r U[i,j,r] B[i,j,r].
                
                # Construct matrix for U
                # We want to find U such that for all i,j: <U[i,j,:], B[i,j,:]> = 1
                # This is equivalent to solving:
                # A_U * U = b, where A_U is (n*m, n*m*r) ? No.
                # Let's flatten U to u (size n*m*r).
                # The constraint is linear in u.
                # For each i,j, the coefficient of u[i*m+j, r] is B[i,j,r].
                # So we have a system: A * u = 1, where A is (n*m, n*m*r).
                # Row (i*m+j) of A has non-zeros at columns (i*m+j)*r + r_idx with value B[i,j,r_idx].
                
                # This matrix A is very sparse and block diagonal in a sense? No.
                # Actually, each row of A corresponds to one (i,j) equation.
                # The variables are U[i,j,r].
                # The equation for (i,j) only involves U[i,j,0..r-1].
                # So the system decouples into n*m independent systems of size 1 x r!
                # sum_{r=0}^{r-1} U[i,j,r] * B[i,j,r] = 1
                # We can solve this for each (i,j) independently.
                # To get a unique solution, we can pick the minimum norm solution:
                # U[i,j,:] = B[i,j,:] / (B[i,j,:] @ B[i,j,:])
                
                B_U = np.zeros((n * m, r))
                for i in range(n):
                    for j in range(m):
                        # B_U[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                        # Vectorized:
                        # V_j = V[j*p:(j+1)*p, :]  (p x r)
                        # W_i = W[i::n, :]         (p x r) -> indices k*n+i for k in 0..p-1
                        # B_U[i*m+j, :] = (V_j * W_i).sum(axis=0)
                        
                        V_j = V[j*p:(j+1)*p, :]
                        W_i = W[i::n, :] # stride n, start i
                        B_U[i*m+j, :] = np.sum(V_j * W_i, axis=0)
                
                # Solve for U
                U_new = np.zeros((n * m, r))
                for idx in range(n * m):
                    b = B_U[idx, :]
                    norm_sq = np.dot(b, b)
                    if norm_sq < 1e-10:
                        # Degenerate, skip or set random
                        U_new[idx, :] = 0.0
                    else:
                        U_new[idx, :] = b / norm_sq
                
                # Solve for W:
                # For each (k,i), sum_r W[k,i,r] * (sum_j U[i,j,r] * V[j,k,r]) = 1
                # Let C[k,i,r] = sum_j U[i,j,r] * V[j*p+k, r]
                # Then sum_r W[k,i,r] * C[k,i,r] = 1
                # Decouples into p*n independent systems.
                
                C_W = np.zeros((p * n, r))
                for k in range(p):
                    for i in range(n):
                        # C_W[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                        # U_i = U[i*m:(i+1)*m, :] (m x r)
                        # V_k = V[::p, k] ? No. V is (m*p, r).
                        # V[j*p+k, r] for j in 0..m-1.
                        # Indices: k, k+p, k+2p, ..., k+(m-1)p
                        V_k = V[k::p, :] # stride p, start k, length m
                        U_i = U[i*m:(i+1)*m, :]
                        C_W[k*n+i, :] = np.sum(U_i * V_k, axis=0)
                        
                W_new = np.zeros((p * n, r))
                for idx in range(p * n):
                    c = C_W[idx, :]
                    norm_sq = np.dot(c, c)
                    if norm_sq < 1e-10:
                        W_new[idx, :] = 0.0
                    else:
                        W_new[idx, :] = c / norm_sq
                
                # Check residual
                # Residual for (i,j,k): sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
                # Compute for a few random points or all points if small enough
                # n*m*p = 84, so check all.
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # Vectorized:
                            u_vec = U_new[i*m+j, :]
                            v_val = V[j*p+k, :]
                            w_val = W_new[k*n+i, :]
                            s = np.sum(u_vec * v_val * w_val)
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                            
                if max_err < 1e-8:
                    # Valid decomposition found
                    return U_new, V, W_new
                
                # Update
                U = U_new
                W = W_new
                
                # Optional: Update V? 
                # If we fix V, we are looking for U,W that work for that V.
                # If no U,W works for random V, maybe try a new V.
                # The alternating minimization above finds the best U,W for fixed V.
                # If the best U,W for a random V doesn't work, it's likely rank r is too low 
                # OR the random V was bad.
                # We can also update V similarly, but U and W are already optimized.
                # Let's just restart with new V if this trial fails.
                break # Break inner loop, try next trial with new V

        return None

    # Search for lower rank
    # We know 84 works. Let's try 83 (Strassen improvement).
    # Then try 82, 81, ... down to some limit.
    # Since 83 is known to exist, we start there.
    
    candidate_ranks = [83, 82, 81, 80, 79, 78, 77, 76, 75]
    
    for r in candidate_ranks:
        if time.time() - start_time > DEADLINE - 10.0:
            break
            
        result = try_rank(r, num_trials=100)
        if result is not None:
            U, V, W = result
            # Final verification
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                # If we found a lower rank, we can stop or try even lower.
                # Given the difficulty, finding 83 is a good win.
                # Let's try a few more if time permits.
                if r == 83:
                    # Try 82 with more trials
                    result = try_rank(82, num_trials=200)
                    if result is not None:
                        U, V, W = result
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                    if abs(s - 1.0) > 1e-8:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            best_rank = 82
                            best_U = U
                            best_V = V
                            best_W = W
                break # Found the best we are looking for in this loop structure
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
