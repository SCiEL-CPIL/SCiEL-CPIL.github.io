"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x3 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # The known minimal rank for 3x3 matrix multiplication is 23 (Landsberg's bound says >= 23, and there exists a rank 23 algorithm)
    # Actually, the exact minimal rank is still an open problem, but it's known to be between 19 and 23.
    # The best known upper bound is 23 (Strassen's algorithm generalized).
    # Wait, let me reconsider. For 3x3 matrix multiplication:
    # - Schoolbook: rank 27
    # - Strassen's algorithm: rank 7 for 2x2, but for 3x3 it's different
    # - The exact minimal rank is unknown, but it's at most 23 (from various constructions)
    # - It's at least 19 (Landsberg's bound)
    
    # Let me try to construct a rank 23 decomposition.
    # Actually, I recall that the best known upper bound for 3x3 matrix multiplication is 23.
    
    # Let me try a different approach: use a known rank-23 algorithm if I can recall it,
    # or try to optimize for lower ranks.
    
    # First, let's try to verify if we can find a rank less than 27.
    # We'll use numerical optimization to find low-rank decompositions.
    
    def matmul_tensor(n, m, p):
        """Create the matrix multiplication tensor T[i,j,k] = delta_{i,k} for the (n x m) @ (m x p) product."""
        # The tensor T is defined such that (A @ B)[i, k] = sum_j A[i, j] * B[j, k]
        # So T[i, j, k] = 1 if the (i,k) entry of A@B depends on A[i,j] and B[j,k]
        # Actually, T is a 3-tensor where T[i, j, k] = 1 means that the product entry (i,k) 
        # gets a contribution from A[i,j] * B[j,k]
        # In our factorization: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] should equal delta_{i,k}... 
        # Wait, no. Let me re-read the convention.
        
        # The convention is W[k*n+i, r]. So the output index is (k, i) where i is row of A, k is col of B.
        # U has shape (n*m, rank): U[i*m+j, r] for A[i,j]
        # V has shape (m*p, rank): V[j*p+k, r] for B[j,k]  
        # W has shape (p*n, rank): W[k*n+i, r] for (A@B)[i,k]
        
        # The tensor equation is: for all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # because (A@B)[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) triple contributes 1 to the product.
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = matmul_tensor(n, m, p)
    
    # Try to find a low-rank decomposition
    # We'll use a numerical approach: minimize the sum of squared errors of the tensor reconstruction
    
    def try_rank(r, max_iters=500, n_restarts=5):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # We want to minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - T[i*m+j, j*p+k, k*n+i])^2
            # But this is a non-convex problem.
            
            # Alternative: use the fact that for a fixed W, the problem in U and V is linear.
            # Or use alternating least squares.
            
            # Let's use a simpler approach: try to find exact solutions with integer/half-integer entries.
            # Since we know the rank is at most 27, let's try ranks 19 through 27.
            
            pass
        
        return None
    
    # Let me try a more direct approach: use known results.
    # The minimal rank for 3x3 matrix multiplication is at most 23.
    # Let me try to construct a rank-23 decomposition.
    
    # Actually, I'll try a numerical optimization approach with careful initialization.
    
    def optimize_rank(r):
        """Optimize for a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        
        # Parameterize U, V, W as flat arrays
        n_params = (n*m + m*p + p*n) * r
        x0 = np.random.randn(n_params) * 0.1
        
        def objective(x):
            if time.time() > deadline - 5:
                return float('inf')
            
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:(n*m + m*p)*r].reshape(m*p, r)
            W = x[(n*m + m*p)*r:].reshape(p*n, r)
            
            # Compute the reconstructed tensor
            # For each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # and compare to T[i*m+j, j*p+k, k*n+i]
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        target = T[i*m+j, j*p+k, k*n+i]
                        error += (val - target) ** 2
            
            return error
        
        # This is too slow for scipy.optimize. Let's use a different approach.
        
        # Use ALS (Alternating Least Squares)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = T[i*m+j, j*p+k, k*n+i] for all k
                    # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) = T[i*m+j, j*p+k, k*n+i]
                    # For fixed i,j, we need to find u = U[i*m+j, :] such that
                    # sum_r u[r] * V[j*p+k,r] * W[k*n+i,r] = T[i*m+j, j*p+k, k*n+i] for all k
                    # This is a linear system: A @ u = b where A[k, r] = V[j*p+k,r] * W[k*n+i,r], b[k] = T[i*m+j, j*p+k, k*n+i]
                    A = np.zeros((p, r))
                    b = np.zeros(p)
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b[k] = T[i*m+j, j*p+k, k*n+i]
                    # Solve least squares
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    b = np.zeros(n)
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b[i] = T[i*m+j, j*p+k, k*n+i]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    b = np.zeros(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b[j] = T[i*m+j, j*p+k, k*n+i]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
        
        # Compute final error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    target = T[i*m+j, j*p+k, k*n+i]
                    error += (val - target) ** 2
        
        if error < 1e-6:
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            error_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                        target = T[i*m+j, j*p+k, k*n+i]
                        error_rounded += (val - target) ** 2
            
            if error_rounded < 1e-6:
                return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from 19 to 27
    for r in range(19, 28):
        if time.time() > deadline - 10:
            break
        
        result = optimize_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we couldn't find a low-rank decomposition, fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
