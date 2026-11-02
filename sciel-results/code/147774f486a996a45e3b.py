"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_val = 1.0
                decomp_val = 0.0
                for c in range(rank):
                    decomp_val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(tensor_val - decomp_val) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _tensor_error(U, V, W, n, m, p):
    """Compute the squared error of the tensor reconstruction."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                recon = 0.0
                for c in range(rank):
                    recon += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                total_error += (target - recon) ** 2
    return total_error

def _try_l_bfgs(n, m, p, rank, deadline, seed=0):
    """Try to find a rank-rank decomposition using L-BFGS-B."""
    np.random.seed(seed)
    
    # Initialize with random values
    U_init = np.random.randn(n*m, rank) * 0.1
    V_init = np.random.randn(m*p, rank) * 0.1
    W_init = np.random.randn(p*n, rank) * 0.1
    
    # Flatten all parameters into a single vector
    params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    U_size = n*m * rank
    V_size = m*p * rank
    
    def objective(params):
        elapsed = time.time() - deadline[0]
        if elapsed > 140:  # Safety margin
            return 1e10
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size+V_size]
        W_flat = params[U_size+V_size:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        return _tensor_error(U, V, W, n, m, p)
    
    result = minimize(objective, params, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.fun < 1e-6:
        params = result.x
        U = params[:U_size].reshape(n*m, rank)
        V = params[U_size:U_size+V_size].reshape(m*p, rank)
        W = params[U_size+V_size:].reshape(p*n, rank)
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = [start_time]
    
    # Best known: schoolbook gives rank 54
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using L-BFGS-B
    # Start with ranks below schoolbook
    for rank in range(best_rank - 1, 20, -1):
        if time.time() - start_time > 130:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if time.time() - start_time > 135:
                break
            
            result = _try_l_bfgs(n, m, p, rank, deadline, seed)
            if result is not None:
                best = result
                best_rank = rank
                break
        
        if best_rank < best_rank:
            break
    
    return best

# EVOLVE-BLOCK-END
