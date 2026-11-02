"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for n×m * m×p."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _compute_error(U, V, W, T):
    """Compute the squared error of the decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                diff = val - T[i, j, k]
                error += diff * diff
    return error

def _round_to_half_integer(x):
    """Round to nearest multiple of 0.5."""
    return round(x * 2) / 2.0

def _try_rank(target_rank, T, time_budget):
    """Try to find a rank-target_rank decomposition using local search."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with small random values
        np.random.seed(42 + restart)
        U = np.random.randn(n * m, target_rank) * 0.3
        V = np.random.randn(m * p, target_rank) * 0.3
        W = np.random.randn(p * n, target_rank) * 0.3
        
        current_error = _compute_error(U, V, W, T)
        
        # Local search
        for iteration in range(5000):
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random column
            c = np.random.randint(target_rank)
            
            # Save current column values
            U_c = U[:, c].copy()
            V_c = V[:, c].copy()
            W_c = W[:, c].copy()
            
            # Try several perturbations
            best_local_error = current_error
            best_local_U = None
            best_local_V = None
            best_local_W = None
            
            for trial in range(5):
                # Random perturbation
                scale = 0.2
                U_trial = U.copy()
                V_trial = V.copy()
                W_trial = W.copy()
                
                U_trial[:, c] += np.random.randn(n * m) * scale
                V_trial[:, c] += np.random.randn(m * p) * scale
                W_trial[:, c] += np.random.randn(p * n) * scale
                
                trial_error = _compute_error(U_trial, V_trial, W_trial, T)
                if trial_error < best_local_error:
                    best_local_error = trial_error
                    best_local_U = U_trial
                    best_local_V = V_trial
                    best_local_W = W_trial
            
            if best_local_U is not None:
                U = best_local_U
                V = best_local_V
                W = best_local_W
                current_error = best_local_error
                
                if current_error < best_error:
                    best_error = current_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            
            # Check convergence
            if current_error < 1e-6:
                break
    
    if best_error < 1e-6 and best_U is not None:
        # Round to half-integers and verify
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in best_U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in best_V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in best_W])
        
        # Verify exactness
        if _compute_error(U_rounded, V_rounded, W_rounded, T) < 1e-9:
            return U_rounded, V_rounded, W_rounded, best_error
    
    return None, None, None, best_error

def _verify_decomposition(U, V, W):
    """Verify exact decomposition."""
    T = _build_tensor()
    error = _compute_error(U, V, W, T)
    return error < 1e-9

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s safety margin
    
    T = _build_tensor()
    rank_schoolbook = n * m * p  # 45
    
    # Schoolbook fallback
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
    
    # Try to find lower rank
    # Try ranks from 20 to 44
    for target_rank in range(20, rank_schoolbook):
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        
        # Allocate time based on how many ranks are left
        time_per_rank = min(5, remaining_time / (rank_schoolbook - target_rank))
        
        U, V, W, err = _try_rank(target_rank, T, time_per_rank)
        
        if U is not None:
            # Found a valid decomposition
            # Convert to list format
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
