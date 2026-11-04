import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

def construct():
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Target tensor T[i, j, k] = 1 for all valid indices
    # The decomposition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Which is equivalent to T[i,j,k] = 1
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """
        Verify that the decomposition is exact (within tol).
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Using einsum for efficiency
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Check if all entries are close to 1
        return np.max(np.abs(reconstructed - 1.0)) < tol

    def objective(params, n, m, p, R):
        """
        Objective function for optimization.
        params: flat array of size (n*m*R + m*p*R + p*n*R)
        """
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size + V_size]
        W_flat = params[U_size + V_size:]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        diff = reconstructed - 1.0
        return np.sum(diff ** 2)

    # Helper to round to nearest 0.5
    def round_to_half(x):
        return np.round(x * 2) / 2

    # Fallback: Schoolbook decomposition (Rank 75)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    # Try to find lower rank decompositions
    # Known minimal rank for 3x5x5 is 69.
    # We try 69, 70, 71...
    
    for target_rank in [69, 70, 71, 72, 73, 74]:
        if time.time() > deadline - 15:  # Keep 15s buffer for fallback/verification
            break
        
        # Define the objective for this specific rank
        def obj_wrapper(params):
            return objective(params, n, m, p, target_rank)
        
        # Try multiple random initializations
        # We use a mix of pure random and "structured" random (perturbed schoolbook)
        num_attempts = 10 if target_rank == 69 else 5
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(123 + attempt * 7 + target_rank)
            
            # Initialization strategy 1: Pure Random
            U_init = np.random.normal(0, 1, (n * m, target_rank))
            V_init = np.random.normal(0, 1, (m * p, target_rank))
            W_init = np.random.normal(0, 1, (p * n, target_rank))
            
            # Initialization strategy 2: Perturbed Schoolbook (if target_rank <= 75)
            if attempt % 2 == 0 and target_rank <= 75:
                # Take first target_rank columns of schoolbook
                U_init = U_school[:, :target_rank].copy()
                V_init = V_school[:, :target_rank].copy()
                W_init = W_school[:, :target_rank].copy()
                # Add noise
                U_init += np.random.normal(0, 0.5, U_init.shape)
                V_init += np.random.normal(0, 0.5, V_init.shape)
                W_init += np.random.normal(0, 0.5, W_init.shape)

            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use L-BFGS-B for large problems
                result = minimize(obj_wrapper, params_init, method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
                
                # Check if we converged to a low error
                if result.fun < 1e-6:
                    # Extract U, V, W
                    U_size = n * m * target_rank
                    V_size = m * p * target_rank
                    W_size = p * n * target_rank
                    
                    U_opt = result.x[:U_size].reshape(n * m, target_rank)
                    V_opt = result.x[U_size:U_size + V_size].reshape(m * p, target_rank)
                    W_opt = result.x[U_size + V_size:].reshape(p * n, target_rank)
                    
                    # Round to nearest 0.5
                    U_round = round_to_half(U_opt)
                    V_round = round_to_half(V_opt)
                    W_round = round_to_half(W_opt)
                    
                    # Verify the rounded version
                    if verify_decomposition(U_round, V_round, W_round, n, m, p, tol=1e-9):
                        best_rank = target_rank
                        best_result = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        # If we found rank 69, we can stop early as it's the known minimum
                        if target_rank == 69:
                            return best_result
                        break # Found valid for this rank, try lower
            
            except Exception:
                continue

    return best_result
