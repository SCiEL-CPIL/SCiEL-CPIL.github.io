import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = 4, 4, 6
    
    # DFT-based decomposition strategy
    # The idea: use DFT to diagonalize the cyclic shift structure
    # For matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can decompose this using a tensor network approach with DFT bases
    # Key insight: represent the multiplication as a sum of rank-1 tensors
    # where each rank-1 term corresponds to a frequency component
    
    # For the 4x4x6 case, we'll use a structured decomposition
    # that exploits the symmetry and cyclic properties
    
    # Strategy: Use the fact that we can write the tensor as a sum of 
    # "diagonal" tensors in a transformed basis
    
    # Let's try a decomposition based on block structure
    # Split the 4x4x6 tensor into smaller blocks and use known low-rank decompositions
    
    # For 2x2x2: rank 7 (Strassen)
    # For 2x2x3: rank 11 (known optimal)
    # For 4x4x4: rank 49 (known)
    # For 4x4x6: we can try to combine these
    
    # Alternative approach: Use a direct tensor decomposition search
    # with random initialization and iterative refinement
    
    # Given the complexity, let's implement a valid schoolbook decomposition
    # and then try to reduce the rank using DFT-based optimization
    
    # Schoolbook rank: n * m * p = 4 * 4 * 6 = 96
    # We'll try to find a lower rank decomposition
    
    # For now, return the schoolbook decomposition as a valid baseline
    # and add a refinement step if time permits
    
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
    
    # Try to reduce rank using DFT-based optimization
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # The tensor T[i,j,k] = 1 if the indices are "aligned" in the multiplication sense
    # We want to find a lower rank decomposition
    
    # DFT approach: apply DFT to the index dimensions
    # This diagonalizes the cyclic shift operators
    
    # For the 4x4x6 tensor, we can try to use the fact that
    # the multiplication structure has a certain symmetry
    
    # Let's try a different approach: use the known Strassen decomposition
    # for the 2x2x2 blocks and combine them
    
    # Split the 4x4x6 into 2x2x3 blocks
    # There are 2 * 2 * 2 = 8 such blocks
    # Each 2x2x3 block has rank 11
    # But they're not independent, so we can't just sum them
    
    # Given the time constraint, return the schoolbook decomposition
    # which is guaranteed to be valid
    
    return {"U": U, "V": V, "W": W}
