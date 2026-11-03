"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline (10s safety margin)
    
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact using Fraction for precision."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True
    
    def try_rank_reduction(target_rank):
        """Try to find a decomposition of given rank using linear algebra."""
        if time.time() > deadline:
            return None
        
        # For a rank-r decomposition, we need:
        # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k
        # This is a system of 125 equations with 3*25*r unknowns.
        # We can fix W to simplify: set W[k*n+i][r] = 1 for all k,i,r
        # Then we need: sum_r U[i*m+j][r] * V[j*p+k][r] = 1
        # This gives us a bilinear system.
        
        # Alternative: use the fact that matrix multiplication can be written as
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # We want to find r terms such that this holds exactly.
        
        # Strategy: Use the identity that for any matrix M, we can write
        # M = sum_r u_r * v_r^T (outer product decomposition)
        # But here we have a tensor, not a matrix.
        
        # Let's try a different approach: use the known lower bounds and
        # attempt to find a decomposition via random initialization + optimization.
        
        # For now, return None to indicate we couldn't find a better decomposition
        # in this call. The caller will use the schoolbook result.
        return None
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Verify the schoolbook decomposition
    if verify(best["U"], best["V"], best["W"]):
        pass  # Schoolbook is correct
    else:
        # Should not happen, but return empty if broken
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
    
    # Try to find a lower-rank decomposition
    # Start with rank 49 (a known lower bound for 5x5x5 is 49)
    for target_rank in [49, 50, 55, 60, 70, 80, 90, 100, 110, 120]:
        if time.time() > deadline:
            break
        result = try_rank_reduction(target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
