"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time

# EVOLVE-BLOCK-START
def construct():
    # We need to find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    # The tensor T has indices (i, j, k) where i in 0..1, j in 0..2, k in 0..3.
    # T[i,j,k] = 1.
    # We want to decompose T as sum_{r=1}^{R} U[i,j,r] V[j,k,r] W[k,i,r].
    # The output factor follows W[k*n+i, r].
    #
    # The schoolbook decomposition has rank 2*3*4 = 24.
    # We can try to find a lower rank decomposition.
    #
    # For small matrix multiplication, known results:
    # - 2x2 by 2x2: rank 7 (Strassen)
    # - 2x3 by 3x4: we need to search.
    #
    # Let's use a random search approach with integer/half-integer constraints.
    # We'll try ranks from a lower bound up to 24, and for each rank, do a random search.
    #
    # The lower bound for rank of matrix multiplication tensor is at least max(n*m, m*p, p*n) = max(6, 12, 8) = 12.
    #
    # We'll use a randomized approach: for a given rank R, initialize U, V, W with random values
    # (integers or half-integers), then try to optimize.
    #
    # Actually, let's think about this differently. The tensor is T[i,j,k] = 1 for all i,j,k.
    # This is a very structured tensor.
    #
    # Consider the following: can we express the product using fewer terms?
    # For 2x3 * 3x4, the output is 2x4 = 8 entries.
    #
    # Let's try a systematic search for small ranks. We'll use a simple approach:
    # For rank R, we need to find U (6xR), V (12xR), W (8xR) with integer/half-integer entries
    # such that for all i,j,k: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    #
    # This is a system of 24 equations (one for each i,j,k) in R*26 unknowns (since U has 6R, V has 12R, W has 8R entries, but they are coupled).
    #
    # Actually, the equations are: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    #
    # Let's try a random search with a time limit.
    
    n, m, p = DIMENSIONS
    deadline = time.time() + 120 - 5  # 115 seconds, leave 5 seconds margin
    
    # We'll try ranks from 12 to 24.
    # For each rank, do a random search.
    
    import random
    import numpy as np
    
    def is_half_integer(x):
        # Check if x is an integer or half-integer (i.e., 2*x is an integer)
        return abs(2*x - round(2*x)) < 1e-9
    
    def verify(U, V, W):
        # Verify the decomposition
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def random_search(rank, max_iters=1000):
        # Random search for a decomposition of given rank
        n, m, p = DIMENSIONS
        random.seed(42)  # Fixed seed for reproducibility
        np.random.seed(42)
        
        for it in range(max_iters):
            # Generate random U, V, W with integer/half-integer entries
            # Use small integers/half-integers to keep values bounded
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                return U, V, W
        return None
    
    # The random search for integer/half-integer solutions is highly unlikely to succeed
    # for rank < 24 in the given time. We return the schoolbook decomposition (rank 24)
    # which is guaranteed to be correct and valid.
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
# EVOLVE-BLOCK-END
