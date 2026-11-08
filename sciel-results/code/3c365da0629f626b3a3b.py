"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # The tensor T has T[i,j,k] = 1 for all i,j,k in the multiplication
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Build the target tensor
    # For each (i,j,k), we need the product to equal 1
    # This is equivalent to: the matrix M_{(i,j),(j',k')} ... 
    # Actually, let's set up the equations properly.
    
    # The condition is: for each (i,k), the (i,k) entry of C = A*B should be sum_j A[i,j]*B[j,k]
    # In the tensor decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k]) * W[k*n+i, c]
    # Wait, that's not right. Let me re-read.
    
    # The tensor decomposition of matrix multiplication means:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where T[i,j,k] = 1 (the standard multiplication tensor)
    # u_c[i*m+j], v_c[j*p+k], w_c[k*n+i]
    
    # So the condition is: for each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 72 equations in 3*rank variables.
    
    # Let's try ranks from 60 down to see if we can find a solution
    # Use a simple iterative approach
    
    for target_rank in range(66, 50, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Define the residual: for each (i,j,k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
        def compute_residual(U, V, W):
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        residual += (val - 1.0) ** 2
            return residual
        
        # Try a few iterations of gradient descent
        lr = 0.01
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
                
            # Compute gradients numerically or analytically
            # d/dU[i*m+j, c] residual = 2 * (val - 1) * V[j*p+k, c] * W[k*n+i, c]
            # But this is for a specific (i,j,k). We need to sum over all (i,j,k) that share the same (i,j) for U.
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = val - 1.0
                        # dU
                        grad_U[i*m+j] += 2 * err * V[j*p+k] * W[k*n+i]
                        # dV
                        grad_V[j*p+k] += 2 * err * U[i*m+j] * W[k*n+i]
                        # dW
                        grad_W[k*n+i] += 2 * err * U[i*m+j] * V[j*p+k]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            res = compute_residual(U, V, W)
            if res < 1e-6:
                break
        
        res = compute_residual(U, V, W)
        if res < 1e-4:
            # Found a good solution, convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if target_rank < best_rank:
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
        
        if time.time() > deadline - 10:
            break
    
    return best_result
# EVOLVE-BLOCK-END
