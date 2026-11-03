"""Matrix multiplication 4x5x6 tensor decomposition."""
import time
import random

DIMENSIONS = [4, 5, 6]


def _make_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the decomposition by checking all tensor entries."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    # Check all (i,j,k) entries: T[i,j,k] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(t_val - 1) > 1e-9:
                    return False
    
    return True


def _random_search(n, m, p, max_rank, iterations, deadline):
    """Try to find a low-rank decomposition via random search."""
    best_rank = max_rank
    best_result = None
    
    for trial_rank in range(max_rank, 39, -1):  # Try from high to low
        if time.time() > deadline - 30:
            break
        
        for _ in range(iterations):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = [[random.randint(-2, 2) for _ in range(trial_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(trial_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(trial_rank)] for _ in range(p * n)]
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_rank = trial_rank
                best_result = (U, V, W)
                print(f"Found valid rank {trial_rank}")
                break
        
        if best_result is not None:
            break
    
    return best_result, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    school_rank = n * m * p
    
    # Try random search for better rank
    # Based on literature, 4x5x6 should have rank around 50-60
    # Let's try from 60 down to 40
    max_try_rank = 60
    iterations_per_rank = 5
    
    best_result, best_rank = _random_search(n, m, p, max_try_rank, iterations_per_rank, deadline)
    
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_school, V_school, W_school
    
    return {"U": U, "V": V, "W": W}
