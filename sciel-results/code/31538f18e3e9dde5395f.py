"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_valid_entries(U, V, W, tol=1e-9):
    """Check that all entries are integers or half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                v2 = 2 * val
                if abs(v2 - round(v2)) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_search(n, m, p, deadline):
    """
    Try to find a low-rank decomposition using structured patterns.
    We'll try ranks from a reasonable lower bound up to schoolbook.
    """
    lower_bound = max(n*m, m*p, p*n)  # 15
    schoolbook = n * m * p  # 45
    
    # Try ranks from lower_bound to schoolbook
    for target_rank in range(lower_bound, schoolbook + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations with structured patterns
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            # Generate random matrices with integer/half-integer entries
            # Use a mix of small values
            U_rand = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
            V_rand = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
            W_rand = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
            
            # Convert to lists
            U_list = [[U_rand[i,j] for j in range(target_rank)] for i in range(n*m)]
            V_list = [[V_rand[i,j] for j in range(target_rank)] for i in range(m*p)]
            W_list = [[W_rand[i,j] for j in range(target_rank)] for i in range(p*n)]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, return schoolbook as fallback
    best_result = _build_schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Strategy: structured search with integer/half-integer entries
    
    np.random.seed(42)
    
    # Try various ranks from lower bound up to schoolbook
    result = _try_structured_search(n, m, p, deadline)
    if result is not None:
        return result
    
    return best_result
# EVOLVE-BLOCK-END
