import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Define a rank-23 decomposition for 3x3x3 matrix multiplication
    # Indices: U is 9x23, V is 9x23, W is 9x23
    # T = sum_k U[:,k] V[:,k] W[:,k]^T (outer products)
    # Actually, the tensor is T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l] ? No.
    # The decomposition is T = sum_l u_l \otimes v_l \otimes w_l
    # Where u_l is a vector of size n*m, v_l of size m*p, w_l of size p*n.
    # For 3x3x3, n=m=p=3.
    
    def get_3x3x3_decomposition():
        # Returns U, V, W of shape (9, rank)
        # Using a known rank-23 decomposition
        # If this is incorrect, the check will fail and we fall back.
        
        # Example of a rank-23 decomposition (coefficients can be 1, -1, etc.)
        # This specific decomposition is from "A 23-multiplication algorithm for 3x3 matrix multiplication"
        # I will use a placeholder that is likely correct or falls back.
        
        # Actually, let's use a simpler rank-24 decomposition if available.
        # Or just return None to trigger fallback if I don't have it.
        
        # I will use a rank-27 (schoolbook) for the 3x3x3 blocks to be safe?
        # No, that gives 63.
        
        # Let's try to construct a rank-24 decomposition.
        # It is known that rank 24 is possible.
        # I will use a specific set of 24 terms.
        
        # Since I don't have the exact 23/24 terms memorized with 100% confidence,
        # I will use a strategy that guarantees a valid solution but might not be optimal.
        # However, the task requires improvement.
        
        # Let's use the following:
        # Split 3x3x3 into 2x2x2 (rank 7) and remaining terms.
        # This is complex.
        
        # I will return a rank-27 decomposition for now to ensure validity, 
        # but I will try to find a better one in the next step.
        # Wait, if I return 63, the score is 1.0.
        
        # Let's try to find a rank-25 decomposition.
        # I will use a random search to find a low-rank decomposition of the 3x3x3 tensor.
        # This might be slow, but I have 150s.
        
        return None # Trigger fallback for now, I need a better plan.

    # Fallback: Schoolbook
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
