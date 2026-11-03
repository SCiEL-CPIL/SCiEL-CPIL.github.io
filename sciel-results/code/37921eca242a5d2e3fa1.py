import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Use einsum for fast calculation
    # U: (n*m, rank) -> reshape to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _get_schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _search_rank(n, m, p, target_rank, time_limit=10):
    """Search for a decomposition of rank target_rank using simulated annealing on discrete values."""
    start_time = time.time()
    
    # Define the discrete value set
    # Half-integers from -2 to 2
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    val_indices = {v: i for i, v in enumerate(values)}
    
    # Initialize U, V, W with random values from the set
    # To improve convergence, we can start with some structure
    # But random is often enough for small problems if we have enough iterations
    
    U = np.random.choice(values, size=(n * m, target_rank))
    V = np.random.choice(values, size=(m * p, target_rank))
    W = np.random.choice(values, size=(p * n, target_rank))
    
    # Reshape for einsum
    U_r = U.reshape(n, m, target_rank)
    V_r = V.reshape(m, p, target_rank)
    W_r = W.reshape(p, n, target_rank)
    
    def calc_error(U, V, W):
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.sum((T - 1.0)**2)
    
    current_error = calc_error(U, V, W)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simulated Annealing
    T_current = 1.0
    T_min = 1e-6
    alpha = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random matrix and a random entry
        mat_idx = np.random.randint(3)
        if mat_idx == 0:
            row, col = np.random.randint(n * m), np.random.randint(target_rank)
            old_val = U[row, col]
            # Pick a new value
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
            U[row, col] = new_val
            new_error = calc_error(U, V, W)
            delta = new_error - current_error
            if delta < 0 or np.random.random() < np.exp(-delta / T_current):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[row, col] = old_val
        elif mat_idx == 1:
            row, col = np.random.randint(m * p), np.random.randint(target_rank)
            old_val = V[row, col]
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
            V[row, col] = new_val
            new_error = calc_error(U, V, W)
            delta = new_error - current_error
            if delta < 0 or np.random.random() < np.exp(-delta / T_current):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[row, col] = old_val
        else:
            row, col = np.random.randint(p * n), np.random.randint(target_rank)
            old_val = W[row, col]
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
            W[row, col] = new_val
            new_error = calc_error(U, V, W)
            delta = new_error - current_error
            if delta < 0 or np.random.random() < np.exp(-delta / T_current):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[row, col] = old_val
        
        T_current *= alpha
        if T_current < T_min:
            T_current = T_min
            
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110  # Leave 10s safety margin
    
    # Get schoolbook as fallback
    schoolbook = _get_schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    # Known lower bound is 12, but likely higher.
    # Let's try ranks 18 to 23.
    # Rank 18 is a known candidate for 2x3x4.
    
    ranks_to_try = [18, 19, 20, 21, 22, 23]
    
    for r in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        # Allocate time for each rank
        # More time for lower ranks as they are harder
        time_per_rank = 15 if r < 20 else 10
        
        result = _search_rank(n, m, p, r, time_limit=time_per_rank)
        
        if result is not None:
            U, V, W = result
            # Verify one more time
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return schoolbook
# EVOLVE-BLOCK-END
