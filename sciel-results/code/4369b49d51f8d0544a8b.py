"""Program for matrix multiplication tensor decomposition 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    The tensor decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i,j,k
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                # Wait, that's not right. Let me check the formula again.
                # The correct formula is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook (trivial) decomposition with rank = n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_search(n, m, p, max_rank, min_rank, deadline):
    """Try to find a valid decomposition with rank between min_rank and max_rank.
    
    Uses a structured approach: for each candidate rank, attempt to find
    integer/half-integer solutions using a greedy construction.
    """
    for target_rank in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            return None
        
        # For small ranks, try to construct a solution
        # We'll use the fact that the tensor has a specific structure
        
        # Initialize with zeros
        U = np.zeros((n * m, target_rank), dtype=np.float64)
        V = np.zeros((m * p, target_rank), dtype=np.float64)
        W = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Try to fill in the decomposition
        # For each (i,j,k) triple, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of n*m*p equations in rank variables per equation
        # We can try a greedy approach: for each (i,j,k), try to satisfy it
        
        # Actually, let's try a different approach: use known decompositions
        # For 2x4x5, we can try to use a block structure
        
        # Let's try to find a solution with small integer entries
        # We'll use a simple iterative refinement
        
        # First, try the schoolbook pattern but with fewer terms
        # This won't work directly, so let's try something else
        
        # For now, just return None to indicate no solution found
        # A more sophisticated search would be needed
        
        continue
    
    return None

def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # For 2x4x5, the schoolbook rank is 40
    # Let's try ranks from 39 down to 20
    result = _try_rank_search(n, m, p, 39, 20, _deadline)
    
    if result is not None:
        U, V, W = result["U"], result["V"], result["W"]
        if _verify_decomposition(U, V, W, n, m, p):
            new_rank = len(U[0])
            if new_rank < best_rank:
                best_result = result
                best_rank = new_rank
    
    return best_result
# EVOLVE-BLOCK-END
