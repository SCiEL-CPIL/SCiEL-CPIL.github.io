import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Build the matrix multiplication tensor T with entries T[i, j, k] = 1 if (i,j,k) is a valid index
    # The tensor is such that C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to find rank-r decomposition: T = sum_{t=0}^{r-1} u_t ⊗ v_t ⊗ w_t
    # where u_t is n*m dimensional, v_t is m*p dimensional, w_t is p*n dimensional
    
    # The tensor T has shape (n*m, m*p) for each "layer" in the third dimension... 
    # Actually, the standard formulation: the matrix multiplication tensor is a 3-tensor
    # T[i,j,k] = 1 if indices are valid, else 0
    # A rank-r decomposition gives:
    #   C[i,k] = sum_t (sum_j A[i,j]*u_t[i*m+j]) * (sum_k B[j,k]*v_t[j*p+k]) * w_t[k*n+i]
    # Wait, let me be more careful.
    
    # Standard formulation: We want U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # For all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # (since the matrix multiplication tensor has 1s at all valid (i,j,k) positions)
    
    # This is a bilinear constraint. Let's use the flattening approach:
    # Fix k, then we have a matrix M_k of size (n*m) x (m*p) where M_k[i*m+j, j*p+k] = 1
    # and we need M_k = U[:, :r] * (something involving V and W)
    
    # Actually, the standard approach: the tensor T can be seen as a matrix of size (n*m) x (m*p) 
    # for each "slice" but that's not quite right either.
    
    # Let me use the direct approach: 
    # T is a 3-tensor of shape (n, m, p) with T[i,j,k] = 1 for all valid i,j,k.
    # A rank-r decomposition: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # For a fixed pair (i,k), we have: 1 = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for all j
    # This is a system of bilinear equations.
    
    # Alternative: use the fact that the tensor can be "flattened" in different ways.
    # Flatten along the first mode: T_{(1)} is a matrix of size (n) x (m*p)
    # Flatten along the second mode: T_{(2)} is a matrix of size (m) x (n*p)
    # Flatten along the third mode: T_{(3)} is a matrix of size (p) x (n*m)
    
    # For our purposes, let's use a numerical optimization approach.
    # We'll try ranks from 25 (theoretical lower bound) upward and see if we can find a valid decomposition.
    
    # The constraints are: for all i in range(n), j in range(m), k in range(p):
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # This is a system of n*m*p = 75 bilinear equations in r*(n*m + m*p + p*n) = r*(15 + 25 + 15) = 55r unknowns.
    
    # For r = 25: 55*25 = 1375 unknowns, 75 equations. Should be feasible.
    
    # Let's use a simple iterative approach: initialize randomly, then use gradient descent or
    # alternating least squares.
    
    # Actually, let's use a simpler approach: since the constraints are bilinear, we can use
    # the "tensor ring" or "CP decomposition" approach with alternating minimization.
    
    # But for small dimensions, let's just try to solve it directly using a numerical method.
    
    # Approach: 
    # 1. Initialize U, V, W randomly
    # 2. For each (i,j,k), the constraint is: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # 3. Use alternating minimization: fix two, solve for the third
    
    # Actually, for this specific problem, let me try a different approach:
    # The matrix multiplication tensor for (3,5,5) might have a known low-rank decomposition.
    
    # Let me just implement a clean schoolbook decomposition first, then try to optimize.
    
    # Schoolbook: rank = n*m*p = 75
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Now let's try to reduce the rank using a numerical optimization approach.
    # We'll use alternating least squares for the CP decomposition of the tensor.
    
    # The tensor T has shape (n, m, p) with T[i,j,k] = 1 for all valid indices.
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k] = 1
    
    # This is equivalent to: for each t, define u_t (n x m), v_t (m x p), w_t (p x n)
    # such that sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1 for all i,j,k
    
    # Let's try to find a lower-rank decomposition using ALS.
    
    def try_rank(r, max_iter=100, tol=1e-6):
        """Try to find a rank-r decomposition using alternating least squares."""
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.zeros((p*n, r))
        
        for t in range(r):
            # Initialize W
            for i in range(n):
                for k in range(p):
                    W[k*n + i, t] = 1.0 / (n*m*p)
        
        for iteration in range(max_iter):
            # Update U: for each (i,j), U[i*m+j, :] should satisfy:
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # For fixed i,j, we need: sum_t U[i*m+j, t] * (V[j*p+k, t] * W[k*n+i, t]) = 1 for all k
                    # Let M[k, t] = V[j*p+k, t] * W[k*n+i, t]
                    # Then: U[i*m+j, :] @ M.T = ones(p)
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve: U[i*m+j, :] @ M.T = ones(p)
                    # This is: M @ U[i*m+j, :].T = ones(p)
                    try:
                        U[i*m+j, :], _ = nnls(M, np.ones(p))
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k, :] should satisfy:
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], _ = nnls(M, np.ones(n))
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, :] should satisfy:
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all j
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], _ = nnls(M, np.ones(m))
                    except:
                        pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < tol:
                return U, V, W, max_err
        
        return U, V, W, max_err
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # 25
    found = False
    for r in range(lower_bound, rank + 1):
        if r > 60:  # Don't go too high
            break
        U_r, V_r, W_r, err = try_rank(r)
        if err < 1e-4:
            found = True
            # Round to nearest half-integer if possible
            U_r = np.round(U_r * 2) / 2
            V_r = np.round(V_r * 2) / 2
            W_r = np.round(W_r * 2) / 2
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # If no lower-rank decomposition found, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
