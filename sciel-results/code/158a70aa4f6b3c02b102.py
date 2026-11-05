import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want U(n*m, r), V(m*p, r), W(p*n, r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try various ranks
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try to find lower rank using numerical methods
    # The tensor has 96 entries, all = 1
    # We can try ranks from some lower bound up
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But actually the rank must be at least the max dimension of the flattening
    
    # Let's try using alternating minimization
    for target_rank in range(33, 96):
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + trial * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating least squares
            converged = False
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let M_{i,j,k,c} = V[j*p+k, c] * W[k*n+i, c]
                # Then sum_c U[i*m+j, c] * M_{i,j,k,c} = 1 for all k
                
                # For fixed (i,j), we have p equations in r unknowns
                # U[i*m+j, :] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                # This is a linear system: U[i*m+j, :] @ M_{i,j} = ones(p)
                # where M_{i,j} is p x r matrix with M_{i,j}[k,c] = V[j*p+k,c]*W[k*n+i,c]
                
                for i in range(n):
                    for j in range(m):
                        # Build M: p x r
                        M_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve U[i*m+j, :] @ M_mat = ones(p)
                        # Least squares
                        try:
                            U[i*m+j, :], residuals, rank, sv = np.linalg.lstsq(M_mat.T, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*p+k, :] @ (U[i*m+j,:] * W[k*n+i,:]) = 1 for all i
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :], residuals, rank, sv = np.linalg.lstsq(M_mat.T, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*n+i, :] @ (U[i*m+j,:] * V[j*p+k,:]) = 1 for all j
                for k in range(p):
                    for i in range(n):
                        M_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :], residuals, rank, sv = np.linalg.lstsq(M_mat.T, np.ones(m), rcond=None)
                        except:
                            pass
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0)**2
                
                if error < 1e-10:
                    converged = True
                    break
                
                if time.time() > deadline - 5:
                    break
            
            if converged:
                # Verify
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0)**2
                if error < 1e-8:
                    best_rank = target_rank
                    best_result = (U.copy(), V.copy(), W.copy())
                    found = True
                    break
        
        if found:
            # Check if we can go lower
            continue
    
    # Convert to lists
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
