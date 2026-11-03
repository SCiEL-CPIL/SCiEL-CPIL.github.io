import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110  # Leave margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 60
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition via optimization
    # We'll try ranks from 55 down to 40
    for target_rank in range(55, 39, -1):
        if time.time() - start_time > timeout - 5:
            break
        result = _try_rank(n, m, p, target_rank, timeout - (time.time() - start_time))
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, target_rank, time_budget):
    """Try to find a rank-target_rank decomposition."""
    start = time.time()
    
    # Build the target tensor T[i,j,k] = 1 if it corresponds to a valid multiplication term
    # T is of shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Initialize with random values
    np.random.seed(42)
    U0 = np.random.randn(nm, target_rank) * 0.1
    V0 = np.random.randn(mp, target_rank) * 0.1
    W0 = np.random.randn(pn, target_rank) * 0.1
    
    def objective(params):
        U = params[:nm*target_rank].reshape(nm, target_rank)
        V = params[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
        W = params[2*nm*target_rank:].reshape(pn, target_rank)
        
        # Compute the tensor: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # This is a 3D tensor of shape (nm, mp, pn)
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        error = np.sum((reconstructed - T) ** 2)
        return error
    
    params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    try:
        result = minimize(objective, params0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if time.time() - start > time_budget:
            return None
        
        U_opt = result.x[:nm*target_rank].reshape(nm, target_rank)
        V_opt = result.x[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
        W_opt = result.x[2*nm*target_rank:].reshape(pn, target_rank)
        
        # Check if the solution is close to integer/half-integer
        # Round to nearest half-integer
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        # Verify exactness
        reconstructed = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
        error = np.sum((reconstructed - T) ** 2)
        
        if error < 1e-6:
            # Convert to list format
            U_list = U_round.tolist()
            V_list = V_round.tolist()
            W_list = W_round.tolist()
            
            # Clean up: convert to int if whole number
            def clean(val):
                if abs(val - round(val)) < 1e-9:
                    return int(round(val))
                elif abs(val * 2 - round(val * 2)) < 1e-9:
                    return val
                return val
            
            U_clean = [[clean(v) for v in row] for row in U_list]
            V_clean = [[clean(v) for v in row] for row in V_list]
            W_clean = [[clean(v) for v in row] for row in W_list]
            
            return {"U": U_clean, "V": V_clean, "W": W_clean}
        else:
            return None
            
    except Exception as e:
        return None
# EVOLVE-BLOCK-END
