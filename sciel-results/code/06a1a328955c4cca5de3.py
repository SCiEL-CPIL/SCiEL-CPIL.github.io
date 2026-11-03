import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Build the multiplication tensor T[i,j,k] = 1 if the (i,k) entry of A*B involves A[i,j]*B[j,k]
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta for each (i,j,k)
    # Actually the tensor is: for each (i,j,k), the coefficient in (A*B)[i,k] from A[i,j]*B[j,k] is 1
    # So T[i,j,k] = 1 for all i,j,k in their ranges
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U is indexed by (i,j), V by (j,k), W by (i,k) [using trace convention W[k*n+i]]
    
    # Target: find rank r factorizations
    # U: (n*m) x r = 9 x r
    # V: (m*p) x r = 12 x r  
    # W: (p*n) x r = 12 x r  [indexed as W[k*n + i, r]]
    
    # The constraint: for all (i,j,k) with i in [0,3), j in [0,3), k in [0,4):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a rank-r decomposition of the 3x3x4 all-ones tensor
    
    # Strategy: Try increasing ranks starting from known lower bounds
    # Known: rank(3x3x3) = 23, so rank(3x3x4) >= 23
    # Upper bound from extension: 23 + 9 = 32, but likely lower
    
    # Let's try to find a solution using the following approach:
    # 1. Start with a known decomposition for 3x3x3 (rank 23)
    # 2. Extend it to 3x3x4
    
    # For 3x3x4, one approach: split into two 3x3x2 problems
    # Or: use the fact that we can "lift" the 3x3x3 solution
    
    # Let me try a direct optimization approach for small ranks
    # Use the fact that this is a bilinear/trilinear form
    
    # Actually, let me try a constructive approach:
    # The 3x3x4 all-ones tensor can be decomposed as:
    # T = T_333 + T_331 where T_333 is the 3x3x3 part (k=0,1,2) and T_331 is k=3
    # T_333 has rank 23, T_331 (3x3x1) has rank 9 (it's just outer products)
    # But we can potentially share factors
    
    # Let me try rank values from 23 upward and solve the system
    def solve_rank(r):
        """Try to find a rank-r decomposition using least squares iteration"""
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # We want: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is equivalent to: the tensor T (shape n,m,p) has rank r decomposition
        
        # Flatten the tensor
        T = np.ones((n, m, p))
        
        # Iterative approach: fix two, solve for the third
        best_err = float('inf')
        for it in range(50):
            if time.time() > deadline:
                break
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: for each k, V[j*p+k,:] * diag(W[k*n+i,:]) @ U[i*m+j,:] = 1
            # Stack over k: matrix A[i,j] where A[i,j][k,r] = sum_s V[j*p+k,s]*W[k*n+i,s]*... 
            # Actually let's be more careful
            
            # For fixed V,W: U[i*m+j,:] is determined by solving:
            # For each k: sum_r U[i*m+j,r] * (V[j*p+k,:] * W[k*n+i,:])_r = 1
            # where (V[j*p+k,:] * W[k*n+i,:])_r = V[j*p+k,r] * W[k*n+i,r]
            
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build the system: p equations, r unknowns
                    # eq_k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve least squares: A_mat @ x = ones(p)
                    if r >= p:
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    else:
                        # Underdetermined, use minimum norm
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    new_U[i*m+j, :] = x
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # eq_i: sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1 for all i
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    new_V[j*p+k, :] = x
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # eq_j: sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1 for all j
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    new_W[k*n+i, :] = x
            W = new_W
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            best_err = min(best_err, err)
            if err < 1e-10:
                break
        
        return U, V, W, best_err
    
    # Try ranks from 23 up
    best_result = None
    for r in range(23, 37):
        if time.time() > deadline:
            break
        U, V, W, err = solve_rank(r)
        if err < 1e-8:
            best_result = (r, U, V, W)
            break
    
    # Fallback to schoolbook if no better found
    if best_result is None:
        r = n * m * p  # 36
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        r, U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(val) for val in row] for row in U]
    V_list = [[float(val) for val in row] for row in V]
    W_list = [[float(val) for val in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
