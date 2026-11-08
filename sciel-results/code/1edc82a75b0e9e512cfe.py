import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try, from lowest known bound upwards
    # Known rank for 3x3 is 21. Let's try 21, 22, 23, 24, 25, 26
    target_ranks = [21, 22, 23, 24, 25, 26]
    
    # Build the target tensor T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # T is indexed by (i, j, k) where i is row of A, j is col of A/row of B, k is col of B
    # The decomposition is C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i, j, k] = 1 for all i, j, k in 0..2
    T = np.ones((n, m, p))
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of this rank
        # We will use random restarts and gradient descent
        found = False
        for attempt in range(50): # Max 50 attempts per rank
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # U: (n, rank), V: (m, rank), W: (p, rank)
            # Note: The problem asks for U of shape (n*m, rank)?? 
            # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
            # Wait, the standard tensor decomposition for matrix multiplication is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # This corresponds to a tensor T[i,j,k].
            # The factors are usually U[i, r], V[j, r], W[k, r].
            # But the interface says:
            # U: (n*m, rank) -> This implies U is flattened?
            # V: (m*p, rank)
            # W: (p*n, rank)
            
            # Let's re-read the interface carefully.
            # "U", "V", "W": factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
            # This is unusual. Standard is (n, rank), (m, rank), (p, rank).
            # If U is (n*m, rank), it might be that the indices are combined?
            # Or maybe the task defines the tensor differently?
            # "output factor follows the official trace convention W[k*n+i,r]"
            # This suggests W is indexed by (k, i).
            
            # Let's look at the seed code:
            # U[i * m + j][column] = 1
            # V[j * p + k][column] = 1
            # W[k * n + i][column] = 1
            
            # So U is indexed by (i, j) flattened to i*m+j.
            # V is indexed by (j, k) flattened to j*p+k.
            # W is indexed by (k, i) flattened to k*n+i.
            
            # The decomposition is:
            # sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This sum must equal T[i, j, k] which is 1 for all i,j,k?
            # No, the tensor for matrix multiplication is T[i, j, k] = 1.
            # The product C[i, k] = sum_j A[i, j] B[j, k].
            # The bilinear form is sum_{i,j,k} A[i,j] B[j,k] C^T[i,k].
            # The tensor T has T[i,j,k] = 1.
            
            # So we need:
            # sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # for all i, j, k in {0, 1, 2}.
            
            # This is a system of 27 equations.
            
            # Let's initialize U, V, W with random values
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # We need to minimize the error:
            # E = sum_{i,j,k} ( sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1 )^2
            
            # This is a non-convex optimization problem.
            # We can use a simple gradient descent or trust region method.
            # Given the small size, we can compute gradients analytically or use finite differences.
            
            # Let's use scipy.optimize.minimize with a custom function.
            # Flatten U, V, W into a single vector x.
            
            def error_func(x):
                U_flat = x[:n*m*rank].reshape(n*m, rank)
                V_flat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                            err += (val - 1.0)**2
                return err
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead or Powell (derivative-free) to avoid gradient computation issues initially
            # Or BFGS with numerical gradients.
            # Since rank is up to 26, variables = 3 * 9 * 26 = 702.
            # Nelder-Mead might be slow.
            # Let's try BFGS.
            
            from scipy.optimize import minimize
            res = minimize(error_func, x0, method='BFGS', options={'maxiter': 2000, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-6:
                # Extract solution
                U_sol = res.x[:n*m*rank].reshape(n*m, rank)
                V_sol = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = res.x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
                
                # Round to integers or half-integers
                # Multiply by 2, round, divide by 2
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
                
                if valid:
                    # Convert to lists of lists as required
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = W_int.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    if best_result:
        return best_result
    else:
        # Fallback to schoolbook
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
