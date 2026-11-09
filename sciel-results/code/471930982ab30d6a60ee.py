import numpy as np
import random
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the schoolbook decomposition as a baseline (rank 75)
    def get_schoolbook():
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
    
    # Verify a decomposition is valid
    def is_valid(U, V, W):
        rank = len(U[0])
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if (i * m + j < n * m and j * p + k < m * p and k * n + i < p * n) else 0
                    # Actually, the tensor entry T[i,j,k] = 1 for all valid i,j,k
                    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        # Also need to ensure that for any (i,j,k) combination that should be 0, it is 0
        # But in matrix multiplication, ALL (i,j,k) with i<n, j<m, k<p contribute to the product
        # The tensor is the multiplication tensor, which has 1s at all positions (i,j,k)
        # So we just need the above check
        return True
    
    # Try to find a lower rank decomposition using random search
    # This is a heuristic approach
    def try_random_search(max_rank, num_trials=100):
        """Try to find a decomposition of rank <= max_rank"""
        for trial in range(num_trials):
            rank = max_rank
            # Initialize with random values
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if is_valid(U, V, W):
                return {"U": U, "V": V, "W": W}
        return None
    
    # First, return schoolbook as guaranteed valid solution
    schoolbook = get_schoolbook()
    
    # Try to find better solutions (this may not succeed within time limit)
    # For now, just return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
