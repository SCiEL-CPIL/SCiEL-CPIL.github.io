"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Strategy: Decompose 3x3x7 into 3x3x3 core + 3x3x4 extension
    # Use schoolbook for core (rank 27) and try to find a low-rank decomposition for extension
    # Total rank = 27 + rank_ext
    
    # The tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # has T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where i indexes rows of A and C, j indexes cols of A / rows of B, k indexes cols of B and C
    # 
    # In the factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # For the full 3x3x7 problem, we can think of it as:
    # Core: i,j,k all in {0,1,2} -> 3x3x3 tensor
    # Extension: i,j in {0,1,2}, k in {3,4,5,6} -> 3x3x4 tensor
    
    # The schoolbook rank is 3*3*7 = 63.
    # We want to do better.
    
    # Approach: Use a randomized search to find a low-rank decomposition.
    # Start with schoolbook and try to remove terms or find linear dependencies.
    
    # Actually, let me use a more systematic approach.
    # The matrix multiplication tensor for n x m x p has a known structure.
    # For 3x3x3, the rank is 23 (known result).
    # For 3x3x4, I'll try to find a good rank.
    
    # Let me use a greedy randomized approach:
    # 1. Start with schoolbook (rank 63)
    # 2. Try to find subsets of terms that are linearly dependent and remove them
    
    # Actually, a better approach: use the known rank-23 decomposition for 3x3x3
    # and find a good decomposition for 3x3x4.
    
    # For 3x3x3, here's a known rank-23 decomposition:
    # I'll use a construction based on Strassen-like ideas.
    
    # Let me just use a safe approach: schoolbook for the whole thing, but verify it works.
    # Then try to improve.
    
    # Schoolbook decomposition:
    rank = n * m * p  # 63
    U = [[0] * rank for _ in range(n * m)]  # 9 x 63
    V = [[0] * rank for _ in range(m * p)]  # 21 x 63
    W = [[0] * rank for _ in range(p * n)]  # 21 x 63
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify the schoolbook works
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook, only one term contributes to each (i,j,k), so T[i,j,k] = 1 for all valid indices
    
    # Now try to improve: use a known rank-23 decomposition for the 3x3x3 core
    # and a rank-30 decomposition for the 3x3x4 extension (if we can find one)
    
    # For now, let me try a randomized search to reduce the rank.
    # I'll try to find a subset of the schoolbook terms that still spans the tensor.
    
    # Actually, let me use a different approach: 
    # The 3x3x3 core has rank 23. Let me construct that explicitly.
    # For the 3x3x4 extension, I'll use schoolbook (rank 36) for now.
    # Total: 23 + 36 = 59, which is better than 63.
    
    # Here's a known rank-23 decomposition for 3x3x3:
    # I'll use a construction from the literature.
    
    # The 23 terms for 3x3x3 (indices: u=i*3+j, v=j*3+k, w=k*3+i for i,j,k in {0,1,2}):
    core_terms = [
        # (u_idx, v_idx, w_idx, coeff)
        # I'll use a standard construction
        (0, 0, 0, 1),    # A[0,0]*B[0,0] -> C[0,0]
        (0, 3, 1, 1),    # A[0,0]*B[1,0] -> C[1,0]  (v=1*3+0=3, w=1*3+0=3? No, w=k*3+i)
        # Wait, let me be more careful with the indexing.
    ]
    
    # Let me use a simpler approach. I'll just use the schoolbook for now,
    # which gives rank 63. This is correct but not optimal.
    
    # To improve, let me try a randomized local search:
    # Start with schoolbook, then try to replace pairs of terms with a single term.
    
    # Actually, for this task, let me just return the schoolbook decomposition
    # which is guaranteed to be correct. The rank is 63.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
