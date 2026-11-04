"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-48 schoolbook decomposition for 3x4 * 4x4 matrix multiplication.
    
    This is the standard naive algorithm.
    Rank = n * m * p = 3 * 4 * 4 = 48.
    
    For each term r corresponding to (i, j, k):
    U[i, j] = 1
    V[j, k] = 1
    W[i, k] = 1
    
    This exactly computes C[i, k] = sum_j A[i, j] B[j, k].
    """
    n, m, p = 3, 4, 4
    total_rank = n * m * p  # 48
    
    U = [[0] * total_rank for _ in range(n * m)]  # (12, 48)
    V = [[0] * total_rank for _ in range(m * p)]  # (16, 48)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 48)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set U
                U[i * m + j][col] = 1
                # Set V
                V[j * p + k][col] = 1
                # Set W (index k*n + i)
                W[k * n + i][col] = 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify correctness with random test
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C_direct = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0] * p for _ in range(n)]
    for r in range(total_rank):
        # Calculate (A . U_r) which is a scalar for this specific sparse U
        # Since U_r has only one 1 at (i,j), A . U_r = A[i,j]
        # But to be generic and safe, we compute the dot product
        sA = 0.0
        for idx in range(n * m):
            if U[idx][r] != 0:
                ii = idx // m
                jj = idx % m
                sA += U[idx][r] * A[ii][jj]
        
        sB = 0.0
        for idx in range(m * p):
            if V[idx][r] != 0:
                jj = idx // p
                kk = idx % p
                sB += V[idx][r] * B[jj][kk]
        
        # Add contribution to C
        for idx in range(p * n):
            if W[idx][r] != 0:
                kk = idx // n
                ii = idx % n
                C_decomp[ii][kk] += W[idx][r] * sA * sB
    
    for i in range(n):
        for k in range(p):
            if C_direct[i][k] != C_decomp[i][k]:
                raise ValueError(f"Mismatch at C[{i},{k}]: direct={C_direct[i][k]}, decomp={C_decomp[i][k]}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
