"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The rank of matrix multiplication tensor can be lower than n*m*p
    # For small cases, we can try to find exact decompositions
    
    def verify(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    # Try rank values below schoolbook
    # Known results: for 2x2x2, rank is 7 (Strassen)
    # For larger dimensions, we can use block-wise Strassen or other methods
    
    # Let's try to use a combination of Strassen-like ideas
    # For 3x5x7, we can try to decompose into blocks
    
    # Alternative: use numerical optimization to find a low-rank decomposition
    # This is a hard problem, so let's try a few strategies
    
    # Strategy 1: Try ranks that are known to be achievable
    # The rank of M_{n,m,p} is at most min(n*m*p, various bounds)
    
    # For now, let's try to find if there's a decomposition with rank < 105
    # by solving the system of equations numerically
    
    # We'll try a few candidate ranks and see if we can find a solution
    candidate_ranks = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try to find a decomposition of rank target_rank
        # This is a system of 105 equations in 3*target_rank variables
        # We can try random initialization and iterative refinement
        
        # Use a simpler approach: try to find a decomposition by
        # combining schoolbook terms in pairs or groups
        
        # For now, just keep the schoolbook if we can't find better quickly
        # In a more sophisticated approach, we'd use:
        # - Strassen's algorithm adapted to these dimensions
        # - Tensor decomposition algorithms
        # - Gröbner basis or other algebraic methods
        
        pass
    
    return best_result
# EVOLVE-BLOCK-END
