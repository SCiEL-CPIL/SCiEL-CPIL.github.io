"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Target: find minimal rank r such that there exist U (nm x r), V (mp x r), W (pn x r)
    # with integer/half-integer entries satisfying:
    # for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school
    
    # Try to find lower rank decompositions using numerical optimization
    # Then rationalize to half-integers
    
    def residual(params, r):
        """Compute residual of tensor equations given flattened params and rank r."""
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k]) # Wait, this is wrong
                    # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # This is not a simple dot product of U and V
                    pass
        return res
    
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a third-order tensor contraction, not a simple matrix product.
    
    # Let me reformulate: define T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need T[i,j,k] = 1 for all i,j,k.
    
    # This is a nonlinear system. Let's use a different approach:
    # Fix U and V, solve for W linearly? No, it's trilinear.
    
    # Alternative: use alternating minimization or gradient descent.
    
    # For a given r, let's try to find U, V, W using gradient descent on the loss:
    # L = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    for r in range(40, best_rank):
        if time.time() > deadline:
            break
        
        # Initialize randomly
        np.random.seed(42 + r)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Gradient descent
        lr = 0.01
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        t_val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                        err = t_val - 1.0
                        # dL/dU[i*m+j,r] = err * V[j*p+k,r] * W[k*n+i,r]
                        dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                        dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                        dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 50 == 0:
                # Check residual
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t_val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                            max_err = max(max_err, abs(t_val - 1.0))
                if max_err < 1e-8:
                    break
        
        # Check if we found a good solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                    max_err = max(max_err, abs(t_val - 1.0))
        
        if max_err < 1e-6:
            # Try to rationalize to half-integers
            U_rat = np.round(U * 2) / 2
            V_rat = np.round(V * 2) / 2
            W_rat = np.round(W * 2) / 2
            
            # Check if rationalized version is still exact
            max_err_rat = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t_val = np.dot(U_rat[i*m+j] * V_rat[j*p+k], W_rat[k*n+i])
                        max_err_rat = max(max_err_rat, abs(t_val - 1.0))
            
            if max_err_rat < 1e-10:
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U_rat]
                V_list = [[float(x) for x in row] for row in V_rat]
                W_list = [[float(x) for x in row] for row in W_rat]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if time.time() > deadline:
            break
    
    return best_result
# EVOLVE-BLOCK-END
