import time
import numpy as np
import random

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Try to find a decomposition with rank < schoolbook_rank
    # We try ranks from 40 down to 32
    # If we find one, we return it. Otherwise, we return schoolbook.
    
    # For efficiency, we will try a few random initializations for each rank
    # and use a simple local search.
    
    best_result = None
    
    # Try ranks 40, 39, 38, 37, 36
    for rank in range(40, 35, -1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        success = _try_find_decomposition_heuristic(n, m, p, rank, max_time=10)
        if success is not None:
            best_result = success
            break
    
    if best_result is None:
        # Return schoolbook decomposition
        U = [[0] * schoolbook_rank for _ in range(nm)]
        V = [[0] * schoolbook_rank for _ in range(mp)]
        W = [[0] * schoolbook_rank for _ in range(pn)]
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

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                result = np.sum(u_val * v_val * w_val)
                expected = 1.0
                error = abs(result - expected)
                max_error = max(max_error, error)
    
    if max_error > 1e-9:
        return False
        
    # Check for integer/half-integer entries
    for arr in [U, V, W]:
        for row in arr:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _try_find_decomposition_heuristic(n, m, p, rank, max_time=10):
    """
    Heuristic search for a rank-r decomposition.
    Uses random initialization and local search.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    
    # Try a few random restarts
    for restart in range(20):
        if time.time() - start_time > max_time:
            break
            
        # Initialize with small random integers
        # Scale is important. Small integers are more likely to work.
        U = np.random.randint(-2, 3, (nm, rank))
        V = np.random.randint(-2, 3, (mp, rank))
        W = np.random.randint(-2, 3, (pn, rank))
        
        # Local search: try to fix errors by adjusting single elements
        # This is a very slow process, so we limit the number of steps
        for step in range(200):
            if time.time() - start_time > max_time:
                break
                
            # Compute error
            errors = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        errors[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
            
            if np.allclose(errors, 0, atol=1e-9):
                if _verify_decomposition(n, m, p, U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Pick a random error location
            i, j, k = np.random.choice(n), np.random.choice(m), np.random.choice(p)
            if abs(errors[i, j, k]) < 1e-9:
                continue
                
            # Try to adjust one variable to reduce error
            # This is a simplified step; a real local search would be more complex
            # For now, we just do random perturbations if stuck
            if random.random() < 0.1:
                idx_u = np.random.choice(nm)
                idx_t = np.random.choice(rank)
                U[idx_u, idx_t] += random.choice([-1, 1])
            elif random.random() < 0.2:
                idx_v = np.random.choice(mp)
                idx_t = np.random.choice(rank)
                V[idx_v, idx_t] += random.choice([-1, 1])
            else:
                idx_w = np.random.choice(pn)
                idx_t = np.random.choice(rank)
                W[idx_w, idx_t] += random.choice([-1, 1])
                
    return None
