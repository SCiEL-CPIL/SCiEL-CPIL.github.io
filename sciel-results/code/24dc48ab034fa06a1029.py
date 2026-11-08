"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Known rank-23 decomposition for 3x3x3 matrix multiplication
    # Based on extensions of Strassen's algorithm
    # Using half-integer coefficients
    
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the rank-23 decomposition explicitly
    # Each entry is (U_index, V_index, W_index, coefficient)
    # U index: i*m + j (0..8)
    # V index: j*p + k (0..8)
    # W index: k*n + i (0..8)
    
    # The decomposition uses the following pattern based on known results
    # for 3x3 matrix multiplication with rank 23
    
    # Using the construction from the literature on small matrix multiplication
    # We define each rank-1 term explicitly
    
    # Term 0: (0,0,0) * 1
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # Term 1: (1,0,0) * 1
    U[1, 1] = 1; V[0, 1] = 1; W[0, 1] = 1
    
    # Term 2: (2,0,0) * 1
    U[2, 2] = 1; V[0, 2] = 1; W[0, 2] = 1
    
    # Term 3: (0,1,0) * 1
    U[3, 3] = 1; V[1, 3] = 1; W[0, 3] = 1
    
    # Term 4: (1,1,0) * 1
    U[4, 4] = 1; V[1, 4] = 1; W[0, 4] = 1
    
    # Term 5: (2,1,0) * 1
    U[5, 5] = 1; V[1, 5] = 1; W[0, 5] = 1
    
    # Term 6: (0,2,0) * 1
    U[6, 6] = 1; V[2, 6] = 1; W[0, 6] = 1
    
    # Term 7: (1,2,0) * 1
    U[7, 7] = 1; V[2, 7] = 1; W[0, 7] = 1
    
    # Term 8: (2,2,0) * 1
    U[8, 8] = 1; V[2, 8] = 1; W[0, 8] = 1
    
    # Terms 9-16: Strassen-like combinations
    # Using the known rank-23 construction
    
    # Let me use a different, verified approach:
    # The rank-23 decomposition for 3x3 matrices is known to exist.
    # I'll use a construction based on the following idea:
    # Split 3x3 into blocks and use Strassen-like combinations
    
    # Actually, let me use a simpler verified construction.
    # The minimal rank for 3x3x3 is known to be 19 (by Coppersmith-Winograd bounds)
    # but explicit constructions achieving < 27 are complex.
    
    # For now, let me verify if rank 23 is achievable with a known explicit form.
    # If not, I'll fall back to a better schoolbook variant or keep 27.
    
    # Let me try a different approach: use the known fact that
    # R(3,3,3) <= 23 has been proven, but I need explicit coefficients.
    
    # Since I cannot guarantee the correctness of an unverified rank-23 decomposition,
    # and the evaluation requires EXACT decomposition, I should be careful.
    
    # Let me verify the schoolbook approach is correct and consider if there's
    # a safe improvement.
    
    # For safety, let me use the schoolbook rank-27 which is guaranteed correct.
    rank = 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
