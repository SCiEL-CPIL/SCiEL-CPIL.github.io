import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed))
    V_seed = np.zeros((m * p, rank_seed))
    W_seed = np.zeros((p * n, rank_seed))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Try to find lower rank
    # Start from rank_seed - 1 down to some lower bound
    # Lower bound for 2x5x5 is at least max(n*m, m*p, p*n) = 25?
    # Actually, rank >= max(nm, mp, pn) is a lower bound? No.
    # Rank >= max(n, m, p) is a lower bound?
    
    for r in range(rank_seed - 1, 24, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random restarts for this rank
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # ALS iterations
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update W: Fix U, V.
                # Eq: sum_c U_ij^c V_jk^c W_ki^c = 1
                # For each (k,i), we have sum_c (U_ij^c V_jk^c) W_ki^c = 1
                # This is a linear system in W_ki.
                # Let A_w be a matrix where rows correspond to (i,j,k) equations?
                # No, the equations are coupled.
                # We have n*m*p equations.
                # Each equation involves W_ki.
                # Let's build the linear system for W.
                # Variables: W_ki for all k,i. Total p*n variables.
                # Equation for (i,j,k): sum_c U_ij^c V_jk^c W_ki^c = 1.
                # This equation ONLY involves W_ki.
                # So for each (k,i), the sum of equations for all (i,j,k) with same (k,i)
                # must equal ... wait.
                # The equation is for EVERY (i,j,k).
                # So for a fixed (k,i), we have m equations (one for each j).
                # sum_c U_ij^c V_jk^c W_ki^c = 1 for all j.
                # This is a system of m equations for the single variable W_ki?
                # No, W_ki is a scalar.
                # So we have m equations:
                # (U_i0^c V_0k^c) W_ki^c + ... = 1
                # (U_i1^c V_1k^c) W_ki^c + ... = 1
                # This overdetermined system for W_ki might not have a solution.
                # We should use least squares.
                
                # Construct matrix for W update
                # Rows: all (i,j,k) triples. Size n*m*p.
                # Cols: all (k,i) pairs. Size p*n.
                # Entry M[row, col] = sum_c U_ij^c V_jk^c if (col corresponds to k,i).
                # Actually, for a specific row (i,j,k) and col (k',i'),
                # M[row, col] is non-zero only if k'=k and i'=i.
                # Then M[row, col] = sum_c U_ij^c V_jk^c.
                
                M_W = np.zeros((n * m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i * m * p + j * p + k
                            col_idx = k * n + i
                            # Sum over c
                            val = np.dot(U[i * m + j, :], V[j * p + k, :])
                            M_W[row_idx, col_idx] = val
                
                # Solve M_W @ W_flat = 1
                # W_flat has size p*n
                try:
                    W_flat, residuals, rank_w, s = np.linalg.lstsq(M_W, np.ones(n * m * p), rcond=None)
                    W = W_flat.reshape(p, n).T # W is p*n, indexed by k*n+i?
                    # W_flat index is k*n+i.
                    # W matrix in output is (p*n, r)? No, W is (p*n, r) in the return dict?
                    # No, W is (p*n, rank).
                    # My W variable here is the coefficient matrix?
                    # Wait, the decomposition is sum_c ...
                    # U, V, W are matrices of shape (size, r).
                    # My ALS update for W should update the columns of W?
                    # No, W in the equation is the vector of coefficients for the term c?
                    # Let's re-read the variable definitions.
                    # U is (nm, r). V is (mp, r). W is (pn, r).
                    # The term is sum_c U_ij^c V_jk^c W_ki^c.
                    # So W_ki^c is the entry at row k*n+i, col c of W.
                    # So for fixed (k,i), we are solving for the vector W[k*n+i, :] of length r.
                    # The equation is:
                    # sum_c (U_ij^c V_jk^c) W_ki^c = 1.
                    # This is (U_ij V_jk)^T @ W_ki = 1.
                    # Let vec_UV = U[i*m+j, :] * V[j*p+k, :] (element-wise product).
                    # Then vec_UV @ W[k*n+i, :] = 1.
                    # We have m such equations for each (k,i) (varying j).
                    # So we form a system:
                    # A @ W_col = 1
                    # Where A is m x r. A[j, c] = U[i*m+j, c] * V[j*p+k, c].
                    # We solve for W_col (size r).
                    
                    # Let's restructure the ALS loop.
                    
                except:
                    pass
            break # Restart loop
            
        # End restarts for r
        
    # Convert to lists of ints/floats
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
