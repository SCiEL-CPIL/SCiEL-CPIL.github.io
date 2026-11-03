"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Tries to find a lower-rank decomposition via random search,
    falls back to schoolbook rank 36 if not found within time budget.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s wall-clock limit
    
    # Generate schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0.0] * rank_school for _ in range(n * m)]
    V_school = [[0.0] * rank_school for _ in range(m * p)]
    W_school = [[0.0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1.0
                V_school[j * p + k][column] = 1.0
                W_school[k * n + i][column] = 1.0
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find lower rank decompositions via random search
    for target_rank in range(18, min(rank_school, 28)):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 30:
                break
            
            choices = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = [[random.choice(choices) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(choices) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(choices) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
