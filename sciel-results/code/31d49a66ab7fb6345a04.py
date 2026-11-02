import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def matmul_residual(params, rank):
        """Compute residual for a given rank decomposition."""
        # params: flattened U, V, W
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Reconstruct tensor: sum_r U[:,r] V[:,r] W[:,r]
        # This is a tensor of shape (n*m, m*p, p*n)
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T - recon)**2)
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from 18 upward
    for rank in range(18, min(best_rank, 25)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_init = np.random.randn(n*m, rank)
            V_init = np.random.randn(m*p, rank)
            W_init = np.random.randn(p*n, rank)
            
            # Scale to help convergence
            U_init *= 0.5
            V_init *= 0.5
            W_init *= 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use alternating minimization instead of full gradient descent
            # This is more stable for tensor decomposition
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each r, U[:,r] = argmin ||T - sum_r' U_r' V_r' W_r'||^2
                # Fix V and W, solve for U
                # The residual is: T - sum_r U[:,r] V[:,r] W[:,r]
                # This is a linear problem in U for fixed V, W
                # Let's use a simpler alternating scheme
                
                # Compute current reconstruction
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T - recon
                
                # Update U: For each column r of U, we want to minimize
                # ||residual + U[:,r] V[:,r] W[:,r] - sum_{r'!=r} U[:,r'] V[:,r'] W[:,r']||^2
                # This is complex. Let's use a simpler approach.
                
                # Actually, let's just do a simple gradient step
                # Compute gradient of residual w.r.t. U, V, W
                
                # grad_U[i,r] = -2 * sum_{j,k} residual[i,j,k] V[j,r] W[k,r]
                # grad_V[j,r] = -2 * sum_{i,k} residual[i,j,k] U[i,r] W[k,r]
                # grad_W[k,r] = -2 * sum_{i,j} residual[i,j,k] U[i,r] V[j,r]
                
                lr = 0.01
                grad_U = -2 * np.einsum('ijk,jr,kr->ir', residual, V, W)
                grad_V = -2 * np.einsum('ijk,ir,kr->jr', residual, U, W)
                grad_W = -2 * np.einsum('ijk,ir,jr->kr', residual, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.linalg.norm(residual) < 1e-10:
                    break
            
            # Check if we found a solution
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - recon)
            
            if err < 1e-6:
                # Try to round to integers/half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                recon_round = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                err_round = np.linalg.norm(T - recon_round)
                
                if err_round < 1e-8:
                    best_result = {
                        'U': U_round.tolist(),
                        'V': V_round.tolist(),
                        'W': W_round.tolist()
                    }
                    best_rank = rank
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Return schoolbook
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
                    
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
