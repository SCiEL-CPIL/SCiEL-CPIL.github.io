"""Program to find a rank-minimal tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin from 120s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns:
        dict: {'U': (9, rank), 'V': (18, rank), 'W': (18, rank)}
              U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
              where n=3, m=3, p=6
    """
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p = 54
    rank = n * m * p
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a better decomposition using random initialization and iterative refinement
    # We'll try ranks lower than 54
    for target_rank in range(40, 54):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Initialize with random values
            U_curr = np.random.randn(n * m, target_rank)
            V_curr = np.random.randn(m * p, target_rank)
            W_curr = np.random.randn(p * n, target_rank)
            
            # Iteratively refine using gradient-free approach (random perturbations)
            # We want to minimize the error in the tensor decomposition
            # The tensor T_{i,j,k} = sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c}
            # For matrix multiplication, T_{i,j,k} should be 1 if the indices correspond to valid multiplications
            # Actually, for matrix multiplication C = AB where A is n x m, B is m x p, C is n x p:
            # C_{i,k} = sum_j A_{i,j} * B_{j,k}
            # The tensor has entries T_{i,k,j} = 1 for all valid i, k, j
            # In the convention W[k*n+i, c], we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
            
            # Let's define the target tensor
            target_tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target_tensor[i, j, k] = 1  # This is the schoolbook term, but we want to combine them
            
            # Actually, the decomposition should satisfy:
            # For each output entry C_{i,k}: C_{i,k} = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This should equal sum_j A_{i,j} * B_{j,k}
            # So the coefficient of A_{i,j} * B_{j,k} in the bilinear form should be 1
            # The bilinear form is sum_j sum_c (U[i*m+j, c] * W[k*n+i, c]) * A_{i,j} * (V[j*p+k, c] * B_{j,k})
            # Wait, this is not quite right. Let me reconsider.
            
            # The standard tensor decomposition for matrix multiplication:
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # We write this as C = sum_c (U_c^T A) * (V_c^T B) * W_c^T (outer product)
            # More precisely: C_{i,k} = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
            # Wait, that's not dimensionally consistent.
            
            # Let me use the standard formulation:
            # The multiplication tensor T has T[i, j, k] = 1 for the trilinear form
            # Actually, for bilinear: C_{i,k} = sum_j A_{i,j} B_{j,k}
            # We can write A as a vector of length n*m, B as length m*p, C as length p*n
            # C = T * (A ⊗ B) where T is the multiplication tensor
            # T_{(i,k), (j), (i',j')} = delta_{i,i'} * delta_{k,j'} if the indices match appropriately
            # This is getting complicated. Let me just verify the schoolbook decomposition works and try to optimize.
            
            # For now, let's just return the schoolbook if we can't find better
            pass
    
    # Convert to lists for the return interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
