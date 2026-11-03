"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Baseline: schoolbook rank 80
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
    
    # Try to find lower rank decomposition
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try random search with small integer/half-integer entries
    # Use a greedy approach: build up the decomposition term by term
    # Each term is a rank-1 tensor: u * v * w
    
    # Strategy: try to find a set of rank-1 tensors that sum to the all-ones tensor
    # Each rank-1 tensor has form: u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # For a rank-1 term to contribute to T[i,j,k], we need u[i*m+j]*v[j*p+k]*w[k*n+i] != 0
    # The sum over all terms must equal 1 for each (i,j,k)
    
    # Let's try a structured approach: use the fact that the tensor is "dense"
    # Try ranks from 60 down to 49
    
    random.seed(42)
    
    def try_random_decomposition(rank, max_iters=5000):
        """Try to find a valid decomposition of given rank using random search."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with random small values
            vals = [-1, -0.5, 0, 0.5, 1]
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try decreasing ranks
    for rank in range(70, 48, -1):
        if time.time() > deadline:
            break
        result = try_random_decomposition(rank, max_iters=2000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
