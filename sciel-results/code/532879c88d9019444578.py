import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to leave margin
    
    # Target tensor T[i,j,k] = 1
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank
    # We search from rank_school - 1 down to some limit
    # Given the small size, we can try a few ranks
    min_rank_to_try = max(1, rank_school - 20) # Try down to 85
    
    for r in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this rank
        found = False
        max_restarts = 10 if r > 100 else 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 100 + r)
            
            # Initialize factors
            # U: (n*m, r)
            # V: (m*p, r)
            # W: (p*n, r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS Loop
            converged = False
            for it in range(100):
                # Update U
                # For each (i,j), solve A u = 1
                # A is p x r, A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Construct A
                        # Rows correspond to k
                        # Cols correspond to c
                        A = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        
                        # Solve A u = ones(p)
                        # Use least squares
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            new_U[i * m + j, :] = u_sol
                        except:
                            pass
                U = new_U
                
                # Update V
                # For each (j,k), solve A v = 1
                # A is n x r, A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            new_V[j * p + k, :] = v_sol
                        except:
                            pass
                V = new_V
                
                # Update W
                # For each (k,i), solve A w = 1
                # A is m x r, A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            new_W[k * n + i, :] = w_sol
                        except:
                            pass
                W = new_W
                
                # Check convergence/residual
                # Compute tensor
                # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            residual += (val - 1)**2
                
                if residual < 1e-10:
                    converged = True
                    break
                    
            if converged:
                # Try to snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify exactness of snapped version
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U_snapped[i * m + j, c] * V_snapped[j * p + k, c] * W_snapped[k * n + i, c]
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_U = U_snapped
                    best_V = V_snapped
                    best_W = W_snapped
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # Convert to list of lists for output
    # If best_rank is still schoolbook, use schoolbook (which is exact integers)
    # If we found a lower rank, use the snapped values
    
    if best_rank == rank_school:
        U_out = [[int(x) for x in row] for row in U_school]
        V_out = [[int(x) for x in row] for row in V_school]
        W_out = [[int(x) for x in row] for row in W_school]
    else:
        # Ensure values are floats that represent half-integers exactly
        # e.g. 0.5, 1.0, 1.5
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        
    return {"U": U_out, "V": V_out, "W": W_out}
