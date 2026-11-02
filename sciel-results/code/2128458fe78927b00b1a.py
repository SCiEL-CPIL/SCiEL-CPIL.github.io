"""Rewritten program for matrix multiplication 2x3x9."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = DIMENSIONS
    r_school = n * m * p  # 54

    # Build the schoolbook decomposition
    U = [[0] * r_school for _ in range(n * m)]
    V = [[0] * r_school for _ in range(m * p)]
    W = [[0] * r_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify the schoolbook decomposition is correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # should equal 1 for all (i,j,k)
    T = np.zeros((n, m, p))
    for r in range(r_school):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    assert np.allclose(T, 1.0), "Schoolbook decomposition verification failed"

    # Try to find a lower rank decomposition using iterative compression
    # Strategy: randomly initialize a lower rank decomposition and try to
    # iteratively adjust it to match the target tensor
    
    # Try ranks below 54
    best_rank = r_school
    best_U = U
    best_V = V
    best_W = W
    
    # For small problems, we can try a few different approaches
    # Approach 1: Try rank 53, 52, etc. with random initialization and gradient descent
    
    # First, let's see if we can find a simple combinatorial improvement
    # For 2x3x9, the tensor has 54 entries. The question is whether any
    # of them can be "absorbed" into the others.
    
    # Try a simple approach: use the schoolbook but see if we can merge
    # some columns by linearity
    
    # Actually, let's try a more systematic approach:
    # Set up the problem as a rank minimization and try to solve it
    
    # For now, return the schoolbook as the baseline
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
