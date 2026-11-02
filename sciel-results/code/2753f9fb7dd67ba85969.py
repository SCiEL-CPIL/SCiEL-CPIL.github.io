import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _solve_w_rational(U, V, n, m, p, r):
    """
    Solve for W over Q.
    Returns W (float array) if a valid solution with small rational entries exists, else None.
    """
    W = np.zeros((p * n, r), dtype=np.float64)
    
    for k in range(p):
        for i in range(n):
            # Build system A x = b
            # A is m x r
            # A[j, t] = U[i, j, t] * V[j, k, t]
            A = np.zeros((m, r), dtype=np.float64)
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            b = np.ones(m, dtype=np.float64)
            
            # Solve A x = b
            try:
                # Check if consistent
                # Use lstsq to get a solution
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is valid (residuals should be 0)
                if np.any(residuals > 1e-9):
                    return None
                
                # Check if solution is "nice" (integer or half-integer)
                # We allow small floating point errors
                scaled = sol * 2.0
                if not np.allclose(scaled, np.round(scaled), atol=1e-6):
                    return None
                
                W[k * n + i, :] = sol
            
            except np.linalg.LinAlgError:
                return None
    
    # Verify over reals
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(s - 1.0) > 1e-6:
                    return None
    
    return W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Trivial decomposition as fallback (Rank 36)
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    random.seed(42)
    
    min_rank_to_try = 24
    max_rank_to_try = 32
    
    for target_rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        max_attempts = 50000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            
            # Random entries in {-1, 0, 1}
            # Density around 0.3-0.5
            for i in range(n * m):
                for t in range(target_rank):
                    r_val = random.random()
                    if r_val < 0.33:
                        U[i, t] = 1
                    elif r_val < 0.66:
                        U[i, t] = -1
                    else:
                        U[i, t] = 0
            
            for i in range(m * p):
                for t in range(target_rank):
                    r_val = random.random()
                    if r_val < 0.33:
                        V[i, t] = 1
                    elif r_val < 0.66:
                        V[i, t] = -1
                    else:
                        V[i, t] = 0
            
            W = _solve_w_rational(U, V, n, m, p, target_rank)
            if W is not None:
                best_result = (target_rank, U, V, W)
                # Convert to list for return
                return {"U": best_result[1].tolist(), "V": best_result[2].tolist(), "W": best_result[3].tolist()}
        
        # If we exhausted attempts for this rank without success, try next rank
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
