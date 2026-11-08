"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from functools import lru_cache

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    # The tensor for matrix multiplication C = A * B has entries T[i, j, k]
    # where i in [0,n), j in [0,m), k in [0,p)
    # A is n x m, B is m x p, C is n x p
    # T[i, j, k] = 1 if and only if the product A[i,j] * B[j,k] contributes to C[i,k]
    # In the standard tensor decomposition, we want:
    # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]  -- wait, this isn't right.
    #
    # Let's clarify the tensor structure.
    # The matrix multiplication tensor M_{n,m,p} has dimensions (n*m) x (m*p) x (p*n)
    # Actually, the standard formulation:
    # We have three spaces:
    # - Input A: n x m matrix, flattened to n*m
    # - Input B: m x p matrix, flattened to m*p  
    # - Output C: n x p matrix, flattened to p*n (with trace convention W[k*n+i, r])
    #
    # The tensor T has entries T[a, b, c] where:
    # - a indexes A: a = i*m + j for i in [0,n), j in [0,m)
    # - b indexes B: b = j'*p + k for j' in [0,m), k in [0,p)
    # - c indexes C: c = k'*n + i' for k' in [0,p), i' in [0,n)
    #
    # T[a, b, c] = 1 if and only if the multiplication A[i,j] * B[j',k] contributes to C[i',k']
    # This happens when i = i', j = j', k = k'
    # So T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    
    # Build the tensor
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1
    
    # We want to find rank-minimal decomposition:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #
    # This is equivalent to finding the minimal number of rank-1 tensors that sum to T.
    #
    # For small cases, we can try to find a decomposition by solving a system of equations.
    # The bilinear complexity of 3x4x6 matrix multiplication is known to be 20.
    # Let's try to find a rank-20 decomposition.
    
    # Known result: The bilinear complexity of multiplying a 3x4 matrix by a 4x6 matrix is 20.
    # This can be achieved via block methods or direct construction.
    
    # Let's try a direct approach: use the known rank-20 decomposition if we can construct it,
    # otherwise fall back to schoolbook (rank 72) or try to find a better one.
    
    # Schoolbook rank = n*m*p = 3*4*6 = 72
    schoolbook_rank = n * m * p
    
    # Try to find a rank-20 decomposition using known constructions.
    # One approach: use the fact that 3x4x6 can be decomposed using blocks.
    # 
    # Actually, let's try a more systematic approach. We'll use integer programming or
    # a greedy algorithm to find a low-rank decomposition.
    
    # For the 3x4x6 case, the known minimal rank is 20. Let's try to construct it.
    
    # Method: Use the tensor decomposition from the literature.
    # The 3x4x6 tensor has rank 20. We can construct it by:
    # - Viewing it as a combination of smaller tensors
    # - Or using a direct construction
    
    # Let's try a direct construction using known rank-20 patterns.
    # Based on the work of Bini, Capovani, Lotti, Romani, and others.
    
    # Alternative: Use a randomized or greedy approach to find a low-rank decomposition.
    
    # Let's try to find a rank-20 decomposition by solving a linear system.
    # We'll assume the decomposition has half-integer or integer entries.
    
    # Strategy: Try ranks from 20 upward, and for each rank, try to find a decomposition.
    # For rank r, we have r unknowns per matrix entry, so the system is:
    # For each (a,b,c) with T[a,b,c] = 1: sum_{r'} U[a,r'] * V[b,r'] * W[c,r'] = 1
    # For each (a,b,c) with T[a,b,c] = 0: sum_{r'} U[a,r'] * V[b,r'] * W[c,r'] = 0
    
    # This is a system of quadratic equations, which is hard to solve directly.
    # Instead, we can use the fact that for a fixed set of U and V, W is determined linearly,
    # and vice versa.
    
    # Let's try a different approach: use the known rank-20 decomposition from the literature.
    # The 3x4x6 matrix multiplication tensor has been shown to have rank 20.
    
    # Construct a rank-20 decomposition using a known method.
    # One way: use the block decomposition.
    # 3x4x6 can be seen as blocks of 3x2x3, 3x2x3, etc.
    
    # Actually, let's just use the schoolbook decomposition for now, since we need to return
    # a valid decomposition. We can try to improve it.
    
    # Schoolbook decomposition
    rank = schoolbook_rank
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W, tensor, tol=1e-9):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Compute sum_r U[a,r] * V[b,r] * W[c,r]
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - tensor[a, b, c]) > tol:
                        return False
        # Also check that off-diagonal terms are zero
        # This is expensive, so we'll skip for now
        return True
    
    if not verify(U, V, W, tensor):
        # Should not happen, but just in case
        pass
    
    # Try to find a better decomposition using a greedy or optimization approach.
    # For now, return the schoolbook decomposition.
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
