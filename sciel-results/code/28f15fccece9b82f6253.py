"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # Target tensor: T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try ranks from lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually a better bound: the tensor has rank at least the max of the "slice ranks"
    # For now, start from a reasonable guess
    
    best_result = None
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try to find better decompositions
    # Use a randomized approach with local search
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, num_trials=50):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            np.random.seed(trial * 1000 + r)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Alternating least squares / coordinate descent
            # For each term c, we can optimize one factor at a time
            # Actually, let's use a different approach: 
            # The constraint is sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            
            # Let's use a gradient-based approach
            # Define the residual tensor R[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
            
            # We want to minimize sum_{i,j,k} R[i,j,k]^2
            
            # Use simple gradient descent with adaptive learning rate
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    return None
                
                # Compute residual
                R = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            R[i,j,k] = sum(U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for c in range(r)) - 1.0
                
                err = np.sum(R**2)
                if err < 1e-12:
                    # Found a solution!
                    U_list = [[float(U[i*m+j,c]) for c in range(r)] for i in range(n) for j in range(m)]
                    V_list = [[float(V[j*p+k,c]) for c in range(r)] for j in range(m) for k in range(p)]
                    W_list = [[float(W[k*n+i,c]) for c in range(r)] for k in range(p) for i in range(n)]
                    return U_list, V_list, W_list
                
                # Compute gradients
                # dR[i,j,k]/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c]
                # dR[i,j,k]/dV[j*p+k,c] = U[i*m+j,c]*W[k*n+i,c]
                # dR[i,j,k]/dW[k*n+i,c] = U[i*m+j,c]*V[j*p+k,c]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r_ik = R[i,j,k]
                            for c in range(r):
                                dU[i*m+j,c] += 2 * r_ik * V[j*p+k,c] * W[k*n+i,c]
                                dV[j*p+k,c] += 2 * r_ik * U[i*m+j,c] * W[k*n+i,c]
                                dW[k*n+i,c] += 2 * r_ik * U[i*m+j,c] * V[j*p+k,c]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                new_err = np.sum((R + lr*dU)**2) if False else err  # simplified
                # Simple: reduce lr if error increases
                if step % 100 == 0:
                    lr *= 0.95
        
        return None
    
    # Try ranks from low to high
    for r in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_trials=30)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END
