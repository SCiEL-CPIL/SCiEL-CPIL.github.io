"""Program for matrix multiplication 3x4x7 tensor decomposition."""
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with trivial decomposition rank = n*m*p
    trivial_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # We'll attempt ranks from trivial_rank-1 down to some minimum
    best_solution = None
    best_rank = trivial_rank
    
    # Precompute the trivial decomposition as starting point
    U_trivial = np.zeros((n*m, trivial_rank))
    V_trivial = np.zeros((m*p, trivial_rank))
    W_trivial = np.zeros((p*n, trivial_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank solutions
    for target_rank in range(trivial_rank - 1, max(trivial_rank - 10, 0), -1):
        # Initialize with first target_rank columns of trivial decomposition
        U0 = U_trivial[:, :target_rank].copy()
        V0 = V_trivial[:, :target_rank].copy()
        W0 = W_trivial[:, :target_rank].copy()
        
        # Define the residual function
        def residual(params):
            total_size = n*m + m*p + p*n
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank:(n*m+m*p)*target_rank]
            W_flat = params[(n*m+m*p)*target_rank:]
            
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Compute the tensor: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            residuals = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        residuals.append(1.0 - val)
            return np.array(residuals)
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if attempt == 0:
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            else:
                # Perturb the initial solution
                noise_scale = 0.1
                U_pert = U0 + noise_scale * np.random.randn(n*m, target_rank)
                V_pert = V0 + noise_scale * np.random.randn(m*p, target_rank)
                W_pert = W0 + noise_scale * np.random.randn(p*n, target_rank)
                x0 = np.concatenate([U_pert.flatten(), V_pert.flatten(), W_pert.flatten()])
            
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=5000)
                if result.cost < 1e-10:
                    # Found a solution, check if it's integer/half-integer
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = result.x[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
                    W_sol = result.x[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to nearest half-integer
                    U_round = np.round(2*U_sol)/2
                    V_round = np.round(2*V_sol)/2
                    W_round = np.round(2*W_sol)/2
                    
                    # Verify the rounded solution
                    def verify(Uv, Vv, Wv):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(Uv[i*m+j, :] * Vv[j*p+k, :] * Wv[k*n+i, :])
                                    if abs(val - 1.0) > 1e-8:
                                        return False
                        return True
                    
                    if verify(U_round, V_round, W_round):
                        best_rank = target_rank
                        best_solution = (U_round, V_round, W_round)
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    # If no better solution found, use trivial
    if best_solution is None:
        U, V, W = U_trivial, V_trivial, W_trivial
    else:
        U, V, W = best_solution
    
    # Convert to lists of lists
    U_list = [[float(U[i,j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i,j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i,j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
