"""Matrix multiplication 3x3x3 - rank-minimal decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin: 130s out of 150s budget
    
    n, m, p = 3, 3, 3
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has shape (n*p, m*p) = (9, 9) in matrix form
    # Actually, the matmul tensor T has indices (i, j, k) where:
    #   i indexes rows of A (and C): 0..n-1
    #   j indexes cols of A / rows of B: 0..m-1  
    #   k indexes cols of B (and C): 0..p-1
    # T[i,j,k] = 1
    # We want to decompose: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i) with row index k*n+i
    # U is shape (n*m, rank) - indexed by (i, j) with row i*m+j
    # V is shape (m*p, rank) - indexed by (j, k) with row j*p+k
    
    # The tensor to decompose: for each (i, j, k) in [0,3)x[0,3)x[0,3):
    #   T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook: rank 27
    U_school = np.zeros((9, 27))
    V_school = np.zeros((9, 27))
    W_school = np.zeros((9, 27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = 27
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find lower rank decompositions using optimization
    # We can use a nonlinear optimization approach
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W form a valid decomposition of the matmul tensor."""
        # For each (i, j, k), verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank, max_iterations=2000, seed=None):
        """Try to find a rank-r decomposition using local optimization."""
        if rank >= best_rank:
            return None
        if time.time() > deadline:
            return None
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(9, rank) * 0.5
        V = np.random.randn(9, rank) * 0.5
        W = np.random.randn(9, rank) * 0.5
        
        # We need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of 27 equations in 3*9*rank variables
        
        # Use a simple gradient descent on the residual
        # f(U,V,W) = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        
        # Actually, let's use scipy's optimization if available
        try:
            from scipy.optimize import minimize
        except ImportError:
            minimize = None
        
        if minimize is not None:
            def objective(params):
                if time.time() > deadline:
                    return float('inf')
                U_p = params[0:9*rank].reshape(9, rank)
                V_p = params[9*rank:18*rank].reshape(9, rank)
                W_p = params[18*rank:27*rank].reshape(9, rank)
                residuals = np.zeros(27)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residuals[idx] = np.dot(U_p[i*m+j], V_p[j*p+k] * W_p[k*n+i]) - 1.0
                            idx += 1
                return np.sum(residuals**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < 1e-10:
                U_opt = result.x[0:9*rank].reshape(9, rank)
                V_opt = result.x[9*rank:18*rank].reshape(9, rank)
                W_opt = result.x[18*rank:27*rank].reshape(9, rank)
                if check_decomposition(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
        
        # Try simple iterative refinement if scipy failed or wasn't available
        # Use alternating least squares style approach
        for iteration in range(300):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For each k, we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    # This is: U[i*m+j,:] @ (V[j*p+k,:]*W[k*n+i,:]).sum over k = 1
                    # Actually it's: for each k separately
                    # Let's think differently. The constraint is:
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                    # This means: U[i*m+j,:] * diag(W[k*n+i,:]) @ V[j*p+k,:]^T ... no
                    # It's element-wise: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                    # So: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
                    # This gives us p=3 equations for U[i*m+j,:]
                    
                    # Collect all constraints for U[i*m+j,:]:
                    # For each k: sum_r U[i*m+j,r] * A_r = 1 where A_r = V[j*p+k,r]*W[k*n+i,r]
                    constraints = np.zeros((p, rank))
                    for k in range(p):
                        constraints[k] = V[j*p+k] * W[k*n+i]
                    # We need U[i*m+j,:] @ constraints.T = ones(p)
                    # This is: constraints @ U[i*m+j,:] = ones(p)
                    try:
                        U[i*m+j] = np.linalg.lstsq(constraints, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    constraints = np.zeros((n, rank))
                    for i in range(n):
                        constraints[i] = U[i*m+j] * W[k*n+i]
                    try:
                        V[j*p+k] = np.linalg.lstsq(constraints, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    constraints = np.zeros((m, rank))
                    for j in range(m):
                        constraints[j] = U[i*m+j] * V[j*p+k]
                    try:
                        W[k*n+i] = np.linalg.lstsq(constraints, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check if we've converged
            if check_decomposition(U, V, W, tol=1e-8):
                # Refine with tighter tolerance
                if check_decomposition(U, V, W, tol=1e-12):
                    return U, V, W
        
        return None
    
    # Try ranks from 19 upward (known lower bound for 3x3x3 is 19)
    # Strassen showed 7 for 2x2x2, and it's known that 3x3x3 has rank <= 23 (Landsberg)
    # and >= 19
    
    for rank in range(19, 27):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            result = try_rank(rank, seed=rank*100 + attempt)
            if result is not None:
                U_opt, V_opt, W_opt = result
                best_rank = rank
                best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    return best_result

# EVOLVE-BLOCK-END
