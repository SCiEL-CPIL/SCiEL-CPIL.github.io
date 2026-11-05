"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # The tensor T has shape (n*m, m*p, p*n) = (20, 30, 24)
    # T[i, j, k] = 1 if i < n, j < m, k < p and the indices correspond to
    # matrix multiplication: A[i,j] * B[j,k] -> C[i,k]
    # In the trace convention W[k*n+i, r], the tensor is:
    # T[a, b, c] where a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 iff the indices are consistent (i.e., there exist i,j,k
    # such that a = i*m+j, b = j*p+k, c = k*n+i)
    
    # Build the tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # The rank of the matrix multiplication tensor is at least max(n*m, m*p, p*n)
    # = max(20, 30, 24) = 30. The schoolbook rank is n*m*p = 120.
    # Known lower bounds: for 4x5x6, the minimum rank is not known exactly,
    # but it's between 30 and 120.
    
    # Strategy: Try to find a low-rank decomposition using a randomized
    # approach with verification. We'll try ranks from a lower bound upward.
    
    # Lower bound: max(n*m, m*p, p*n) = 30
    # Upper bound: n*m*p = 120
    
    # For small problems, we can try a direct approach:
    # Use the fact that the tensor is sparse (only n*m*p = 120 non-zero entries)
    # and try to find a decomposition with rank close to the lower bound.
    
    # One approach: use a greedy algorithm to build the decomposition
    # column by column, choosing columns that maximize the reduction in
    # the residual tensor.
    
    best_rank = n * m * p
    best_decomp = None
    
    # Start with the schoolbook solution as a baseline
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    best_decomp = (U, V, W)
    best_rank = rank_school
    
    # Try to improve using a greedy rank reduction approach
    # We'll try to merge columns where possible
    
    # Alternative: use a randomized tensor decomposition approach
    # Try different ranks and use optimization to find valid decompositions
    
    # For this specific problem, let's try a structured approach:
    # The tensor has a special structure related to matrix multiplication.
    # We can exploit this to find a better decomposition.
    
    # Try rank values from 30 upward, but limit the search
    # to keep within time budget
    min_rank = max(n * m, m * p, p * n)  # 30
    max_try = min(120, min_rank + 40)  # Try up to rank 70
    
    for rank in range(min_rank, max_try + 1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition of this rank
        # Use a simple approach: initialize with random values and verify
        # (This is unlikely to work for exact decomposition, but let's try)
        
        # Actually, for exact integer/half-integer decompositions,
        # we need a different approach. Let's use the structure of the problem.
        
        # The matrix multiplication tensor for (n,m,p) has a known
        # decomposition structure. For small dimensions, we can try
        # to find the minimum rank by exploiting the tensor's symmetry.
        
        # For now, let's stick with the schoolbook if we can't find better.
        # The key insight is that for 4x5x6, the minimum rank might be
        # achievable with a clever decomposition.
        
        # Let's try a specific construction:
        # Split the middle dimension into parts and use block structure.
        
        # Actually, let's just return the best we have.
        # The schoolbook decomposition is guaranteed to work.
    
    # Return the best decomposition found
    U_best, V_best, W_best = best_decomp
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
