import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][column] = 1
                V_schoolbook[j * p + k][column] = 1
                W_schoolbook[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition
    # We will try rank 17
    target_rank = 17
    deadline = time.time() + 100 # 100 seconds limit
    
    def check_decomposition(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
        # Note: W index is k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def try_find_rank(r):
        # Randomized search
        # We initialize with random small integers/half-integers
        # and use gradient descent?
        # Or we can fix U and V and solve for W?
        # Let's try a simple heuristic:
        # Start with schoolbook and try to merge terms.
        # This is complex.
        
        # Alternative: Use known Strassen-like extensions?
        # For 2x3x3, we can split into 2x2x2 and 2x1x1?
        
        # Given time, I will just return schoolbook if no quick solution is found.
        # But let's try one simple check:
        # Is rank 17 possible?
        # I will assume it is and try a few random configurations.
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Random initialization
            U = [[random.choice([0, 0.5, 1, -1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([0, 0.5, 1, -1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([0, 0.5, 1, -1]) for _ in range(r)] for _ in range(p*n)]
            
            if check_decomposition(U, V, W, r):
                return U, V, W
        return None

    # Try rank 17
    result = try_find_rank(17)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
