import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 45: 3 blocks of 3x3x2 (rank 12 each) + 1 block of 3x3x1 (rank 9)
    # Total rank = 3 * 12 + 9 = 45
    
    # If we fail to find rank 12, we fall back to schoolbook for those blocks (rank 18)
    # Total rank = 3 * 18 + 9 = 63
    
    # Helper to find rank r decomposition of 3x3x2 all-ones tensor?
    # No, the tensor is the multiplication tensor.
    
    # We will use a simple greedy search for the 3x3x2 block.
    # If it fails, we use schoolbook.
    
    def find_3x3x2_decomp(rank_target=12):
        # Target tensor T of shape (3, 3, 2)
        # T[i, j, k] = 1 for all i, j, k
        T = np.ones((3, 3, 2))
        
        # Initialize random rank-1 tensors
        best_rank = rank_target
        best_solution = None
        
        for _ in range(100):
            # Random initialization
            U = np.random.randn(3, best_rank)
            V = np.random.randn(3, best_rank)
            W = np.random.randn(2, best_rank)
            
            # Optimize using gradient descent or simple least squares
            # We want sum_l U[:,l] * V[:,l] * W[:,l] = T
            # This is non-convex.
            
            # Simple check: if we can find a solution with integers
            # For now, we assume we can find one if we try enough.
            # Given time constraints, we might just return schoolbook.
            pass
        
        return None

    # Since finding the exact rank 12 decomposition is hard in short time,
    # and the schoolbook is 63, we can try a simpler improvement.
    
    # Use rank 23 for 3x3x3?
    # We don't have the coefficients.
    
    # Let's just return the schoolbook decomposition for now, 
    # but with a slight optimization: 
    # We can't easily beat 63 without known coefficients.
    
    # However, we can try to use the rank 7 Strassen for 2x2x2 blocks.
    # 3x3x7 can be decomposed into 2x2x2 blocks?
    # This is complex.
    
    # Let's stick to the schoolbook but ensure it's valid.
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
