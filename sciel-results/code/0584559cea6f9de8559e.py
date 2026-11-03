"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = 110  # seconds, safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For each candidate rank, try to solve for U, V, W
    # The condition is: for all i,j,j',k,k',i':
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = 1 if j==j' and i==i' and k==k' else 0
    
    # This is a system of nonlinear equations. Use random initialization and optimization.
    
    # Try ranks from best_rank-1 down to some minimum
    # For 3x3x3x5, try ranks 40, 35, 30, 25, 20
    
    candidate_ranks = [40, 35, 30, 25, 20]
    
    for rank in candidate_ranks:
        if time.time() - start_time > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - start_time > deadline - 30:
                break
            
            # Random initialization
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Define the objective: sum of squared errors
            def objective(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k in range(p):
                                for k2 in range(p):
                                    for i2 in range(n):
                                        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                                        actual = 0.0
                                        for r in range(rank):
                                            actual += U_flat[i*m+j, r] * V_flat[j2*p+k, r] * W_flat[k2*n+i2, r]
                                        error += (actual - expected) ** 2
                return error
            
            # Use a simple gradient-free method: Nelder-Mead or just random search
            # Given time constraints, use a few iterations of random perturbation
            
            current_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            current_obj = objective(current_params)
            
            for iteration in range(100):
                if time.time() - start_time > deadline - 30:
                    break
                
                # Random perturbation
                step_size = 0.1 / (1 + iteration * 0.01)
                perturbed = current_params + np.random.randn(len(current_params)) * step_size
                new_obj = objective(perturbed)
                
                if new_obj < current_obj:
                    current_params = perturbed
                    current_obj = new_obj
            
            # Check if we found a valid solution
            if current_obj < 1e-6:
                U_sol = current_params[:n*m*rank].reshape(n*m, rank)
                V_sol = current_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = current_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Convert to lists
                U_list = U_sol.tolist()
                V_list = V_sol.tolist()
                W_list = W_sol.tolist()
                
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
        
        if best_rank < n * m * p:
            break
    
    return best
# EVOLVE-BLOCK-END
