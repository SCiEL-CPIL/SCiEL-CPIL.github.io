"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank 28 with W=1 approach
    r = 28
    if time.time() < deadline - 20:
        # Set W[k*n+i, r] = 1 for all
        W = np.ones((p * n, r))
        
        # Now solve: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] = 1
        # This is a linear system in the variables U and V
        # Unknowns: U (n*m x r) and V (m*p x r)
        # Equations: n*m*p = 56
        
        # Let's parameterize: for each r, let u_r be a vector in R^{n*m} and v_r in R^{m*p}
        # Equation for (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] = 1
        
        # This is still nonlinear (product of u and v). But if we fix v, it's linear in u.
        
        # Strategy: Initialize v randomly, solve linear system for u, then update v, iterate
        
        np.random.seed(42)
        V = np.random.randn(m * p, r)
        
        for iteration in range(50):
            if time.time() > deadline - 20:
                break
            
            # Solve for U given V: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] = 1
            # Group by (i,j): for fixed (i,j), we have p equations:
            # sum_r U[i*m+j,r] * V[j*p+k,r] = 1 for k=0,...,p-1
            # This is: U[i*m+j,:] @ V[j*p:(j+1)*p, :] = ones(p)
            # i.e., U[i*m+j,:] @ V_block = ones(1,p) where V_block is p x r
            
            U = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    # Solve U[i*m+j,:] @ V[j*p:(j+1)*p, :] = ones(1, p)
                    V_block = V[j * p : (j + 1) * p, :]  # p x r
                    # We want u such that u @ V_block = ones(1, p)
                    # This is a linear system: V_block^T @ u^T = ones(p, 1)
                    # i.e., V_block^T @ u = ones(p)
                    Vt = V_block.T  # r x p
                    rhs = np.ones(p)
                    # Solve Vt @ u = rhs, where u is r-dimensional
                    if r >= p:
                        # Underdetermined, use least squares
                        u, residuals, rank_u, sv = np.linalg.lstsq(Vt, rhs, rcond=None)
                        U[i * m + j, :] = u
                    else:
                        U[i * m + j, :] = np.linalg.solve(Vt, rhs)
            
            # Check error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Update V: for each (j,k), solve for V[j*p+k,:] given U
            # sum_r U[i*m+j,r] * V[j*p+k,r] = 1 for all i
            # For fixed (j,k): U[:, j-related rows]... 
            # For fixed j,k: sum_r U[i*m+j,r] * V[j*p+k,r] = 1 for i=0,...,n-1
            # i.e., V[j*p+k,:] @ U_block = ones(n) where U_block[i, r] = U[i*m+j, r]
            for j in range(m):
                for k in range(p):
                    U_block = np.array([U[i * m + j, :] for i in range(n)])  # n x r
                    # V[j*p+k,:] @ U_block^T = ones(n)
                    # U_block @ V[j*p+k,:]^T = ones(n)
                    # U_block @ v = ones(n) where v = V[j*p+k,:]
                    Ut = U_block  # n x r
                    rhs = np.ones(n)
                    if r >= n:
                        v, _, _, _ = np.linalg.lstsq(Ut, rhs, rcond=None)
                        V[j * p + k, :] = v
                    else:
                        V[j * p + k, :] = np.linalg.solve(Ut, rhs)
        
        # If we get here, check one more time
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :])
                    max_err = max(max_err, abs(val - 1))
        if max_err < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
