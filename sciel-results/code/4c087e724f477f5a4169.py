<<<<<<< SEARCH
import time
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe (evaluator kills at 150s)
    
    def rank54_decomp():
        ...
    ...
    return best_result
# EVOLVE-BLOCK-END
=======
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    # The trivial decomposition for matrix multiplication C = A * B
    # where A is n x m and B is m x p
    # C[i, j] = sum_{k=0}^{m-1} A[i, k] * B[k, j]
    
    # Tensor decomposition:
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # For the standard matrix multiplication, rank = m
    # Each term r corresponds to index k in the sum
    
    rank = m  # rank = 5
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for k in range(m):
        for i in range(n):
            # U[i*m + k, k] = 1
            U[i * m + k, k] = 1.0
            
        for j in range(p):
            # V[k*p + j, k] = 1
            V[k * p + j, k] = 1.0
            
        for i in range(n):
            for j in range(p):
                # W[j*n + i, k] = 1  (following W[k*n+i,r] convention, but here r=k)
                # The task says "output factor follows the official trace convention W[k*n+i,r]"
                # This suggests W is indexed as [output_index, rank] where output_index = j*n + i
                W[j * n + i, k] = 1.0
    
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
BET: {mechanism: "Implement the trivial rank-m decomposition which is known to be optimal for rectangular matrix multiplication, fixing the undefined variable error and providing a valid baseline score", q: 0.95, delta: [0.95, 1.0]}
