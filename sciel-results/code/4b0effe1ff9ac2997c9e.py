import time
import sys
import random
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a low-rank exact tensor decomposition for matrix multiplication.
    Falls back to schoolbook if no better solution is found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition."""
        rank = schoolbook_rank
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
    
    def verify_decomposition(U, V, W):
        """
        Verify that the decomposition exactly decomposes the matrix multiplication tensor.
        For each (i, j, k), the sum over r of U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] 
        should equal 1 if this is a valid entry, and the decomposition must be exact.
        
        Actually, the tensor decomposition means:
        For all i, j, k: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
        """
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    def try_rank(rank):
        """
        Try to find a rank-r decomposition using randomized search.
        """
        if rank >= schoolbook_rank:
            return schoolbook_decomposition()
        
        # Use random initialization with structured patterns
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random candidate with integer/half-integer entries
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try progressively lower ranks
    best_result = schoolbook_decomposition()
    
    # Start from a reasonable lower bound and try to find better
    # Theoretical lower bound is max(n*p, m*n, m*p) but we can do better
    min_possible = max(n * p, m * n, m * p)
    
    for target_rank in range(min_possible, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_result = result
            break  # Found the lowest rank we tried
    
    # If we haven't found anything better, use schoolbook
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
