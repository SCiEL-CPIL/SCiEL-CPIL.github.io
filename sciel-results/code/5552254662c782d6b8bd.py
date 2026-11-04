"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if val != 1:
                    return False
    return True

def _is_valid_entry(x):
    """Check if entry is integer or half-integer."""
    try:
        x = float(x)
    except:
        return False
    if abs(x - round(x)) < 1e-9:
        return True
    if abs(2*x - round(2*x)) < 1e-9:
        return True
    return False

def _check_all_entries(U, V, W):
    for row in U:
        for x in row:
            if not _is_valid_entry(x):
                return False
    for row in V:
        for x in row:
            if not _is_valid_entry(x):
                return False
    for row in W:
        for x in row:
            if not _is_valid_entry(x):
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

def _try_structured_search(rank, n, m, p, max_attempts=100, seed=42):
    """Try structured random search for a rank-r decomposition."""
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for c in range(rank):
            # Generate sparse column vectors with small entries
            for idx in range(n*m):
                if rng.random() < 0.4:
                    U[idx][c] = rng.choice([-1, 1, -0.5, 0.5])
            for idx in range(m*p):
                if rng.random() < 0.4:
                    V[idx][c] = rng.choice([-1, 1, -0.5, 0.5])
            for idx in range(p*n):
                if rng.random() < 0.4:
                    W[idx][c] = rng.choice([-1, 1, -0.5, 0.5])
        
        if _verify(U, V, W, n, m, p):
            if _check_all_entries(U, V, W):
                return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 25, -1):
        if time.time() > deadline:
            break
        
        result = _try_structured_search(target_rank, n, m, p, max_attempts=50, seed=target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
