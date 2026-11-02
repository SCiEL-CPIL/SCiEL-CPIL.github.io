"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import random

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard schoolbook rank n*m*p decomposition."""
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

def _validate(n, m, p, U, V, W, tolerance=1e-9):
    """
    Check whether (U, V, W) exactly decomposes the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tolerance:
                    return False
    return True

def _try_structured_decomposition(n, m, p, rank):
    """
    Try a structured decomposition where we partition the middle dimension.
    For each column c, we assign a (i,j,k) triple and set U,V,W to 1 for that triple.
    But we can try to share columns across multiple triples if the structure allows.
    
    Actually, for matrix multiplication, each (i,j,k) needs its own column in the
    schoolbook. To get lower rank, we need to find linear dependencies.
    
    Let's try a different approach: use the fact that for fixed i, the sum over j
    and k must satisfy certain conditions.
    """
    # This is complex, so let's try a more direct search with better initialization
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115 second limit, leave 5s margin
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # The known lower bound for 3x4x5 matrix multiplication rank is at least 
    # max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Schoolbook is 60. Let's try to find something better.
    
    # Strategy: For each rank r from best_rank-1 down to 20, try random searches
    # with better initialization and more attempts
    
    min_possible_rank = max(n * m, m * p, p * n)
    
    for rank in range(best_rank - 1, min_possible_rank - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts based on remaining time
        remaining = deadline - time.time()
        max_attempts = min(100, max(5, int(remaining / 0.1)))
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Better initialization: use structured random with some zeros
            # to encourage sparsity
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Initialize with random sparse structure
            for c in range(rank):
                # Each column corresponds to a "tensor slice"
                # Randomly choose which (i,j,k) this column affects
                # But we need to ensure the tensor identity holds
                
                # Simple approach: random values
                for idx in range(n * m):
                    if random.random() < 0.3:
                        U[idx][c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                for idx in range(m * p):
                    if random.random() < 0.3:
                        V[idx][c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                for idx in range(p * n):
                    if random.random() < 0.3:
                        W[idx][c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            
            if _validate(n, m, p, U, V, W, tolerance=1e-6):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, continue to lower ranks
            continue
        else:
            # Didn't find at this rank, but keep trying lower ranks
            # (the search might succeed at lower ranks with different random seeds)
            pass
    
    return best
# EVOLVE-BLOCK-END
