"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline to stay within 150s wall-clock
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 80
    
    # Known result: 4x4x5 matrix multiplication can be done in rank 72
    # This is based on the fact that 4x4x4 can be done in rank 49 (Strassen-like)
    # and we can add 5 extra columns efficiently
    
    # Let's try to construct a rank-72 decomposition
    # Strategy: Use the known 4x4x4 rank-49 decomposition (if we can find one)
    # and extend it to 4x4x5
    
    # Actually, let's think about this differently.
    # The tensor T_{ijk} for (i,j,k) in [0,4)x[0,4)x[0,5)
    # We need to find rank-r decomposition
    
    # A simple approach: use random initialization and verify
    # But we need an exact decomposition
    
    # Let's use a structured approach:
    # For 4x4x5, we can view it as 4x4x4 + 4x4x1
    # 4x4x4 has rank 49 (optimal known), 4x4x1 has rank 16
    # Total: 49 + 16 = 65? But there might be overlaps
    
    # Actually, let's just try to find a valid decomposition with rank < 80
    # using a greedy or structured approach
    
    # Simple valid construction: 
    # Use the standard bilinear algorithm but merge some terms
    
    # Let's try rank 72: we can save 8 terms
    # One approach: for the last column of B (k=4), we can share some computations
    
    # Actually, let's use a concrete construction:
    # We'll build U, V, W matrices of size (16, r), (20, r), (20, r)
    
    # Start with schoolbook and try to reduce
    # For now, let's return a known valid decomposition
    
    # Rank 72 construction:
    # The idea is that for k=0,1,2,3 we use a 4x4x4 decomposition
    # and for k=4 we use a simpler scheme
    
    # Let's use the following approach:
    # We know that 4x4 matrix multiplication can be done in rank 49
    # Let's construct that first, then add the 5th column
    
    # Actually, let's just do a direct construction that we can verify
    
    # For simplicity and correctness, let's use rank 76 or 72
    # with a structured pattern
    
    # Here's a concrete rank-72 construction:
    # We'll use the fact that we can combine some of the bilinear terms
    
    rank = 72
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill in the decomposition
    # We'll use a pattern based on Strassen-like ideas
    # For 4x4x4, use rank 49, for the extra column use rank 23
    # Total: 49 + 23 = 72
    
    # 4x4x4 part (k=0,1,2,3):
    # Use a known 49-term decomposition
    # This is complex, so let's use a simpler valid construction
    
    # Simple valid construction for rank 72:
    # For k in 0,1,2,3: use 16 terms each = 64 terms
    # For k=4: use 8 terms (shared with some of the above)
    # Total: 64 + 8 = 72
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):  # First 4 columns
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Now for k=4, we need 8 more terms
    # We can use a simple scheme:
    for i in range(n):
        for j in range(m):
            if (i + j) % 2 == 0:  # Use only half the terms
                U[i*m + j, col] = 1
                V[j*p + 4, col] = 1
                W[4*n + i, col] = 1
                col += 1
    
    # This gives us 64 + 8 = 72 terms
    # But wait, this doesn't correctly compute the product
    # We need to verify this is correct
    
    # Actually, the above is NOT correct because we're missing terms
    # Let's use a different approach
    
    # Correct approach: just use schoolbook for now and try to optimize
    # But we know rank 72 is achievable
    
    # Let's use a verified construction:
    # The 4x4x5 tensor can be decomposed in rank 72
    # We'll use a known result from the literature
    
    # For now, let's return the schoolbook decomposition (rank 80)
    # and note that we should find a better one
    
    # Actually, let's try a random search for a valid decomposition
    
    # Better approach: use the structure of the problem
    # The matrix multiplication tensor has a lot of symmetry
    
    # Let's just return schoolbook for correctness
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
