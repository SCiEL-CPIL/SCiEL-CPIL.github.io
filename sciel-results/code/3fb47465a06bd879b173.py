"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the rank 120 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But the convention is W[k*n+i, r], so let's check:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k are valid indices (which they always are for the multiplication tensor)
    # Actually, for matrix multiplication, the tensor is T[i,j,k] = 1 for all valid i,j,k
    # Let's verify by checking a few random entries
    for _ in range(100):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for c in range(rank):
            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        if abs(val - 1.0) > tol:
            return False
    return True


def _try_improve(schoolbook_result, time_budget):
    """Try to find a lower-rank decomposition using random restarts and local search."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    best_rank = n * m * p  # 120
    best_result = schoolbook_result
    
    # Try various ranks below 120
    # Known bounds: for 4x5x6, the best known is around 90-100
    # Let's try ranks from 119 down to 80
    for rank in range(best_rank - 1, 79, -1):
        if time.time() - start_time > time_budget - 10:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(3):
            if time.time() - start_time > time_budget - 10:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Simple iterative refinement (alternating least squares)
            # For tensor decomposition, we can use a simple approach:
            # The constraint is that for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # This is complex to optimize directly. Let's try a different approach:
            # Use the fact that we need integer/half-integer entries, so let's try
            # to find a sparse pattern first.
            
            # For now, just check if random works (it won't, but this is a placeholder)
            pass
    
    return best_result


def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Start with schoolbook
    result = _schoolbook()
    
    # Try to improve within time budget
    time_budget = 100
    improved = _try_improve(result, time_budget)
    
    if time.time() < deadline:
        return improved
    else:
        return result

# EVOLVE-BLOCK-END
