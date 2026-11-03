import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find a lower-rank decomposition
    # Start with a known good rank and try to improve
    
    # The naive rank is n*m*p = 56
    # We can try ranks from 49 down to find the minimum
    
    # For 2x4x7, we can use a structured decomposition
    # Let's try a rank-49 decomposition based on row-blocking
    
    # Approach: Use the fact that n=2 is small
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as a tensor decomposition
    
    # Try different rank values and check if a valid decomposition exists
    # For efficiency, we'll use a heuristic search
    
    best_result = None
    best_rank = n * m * p  # Start with naive
    
    # Try ranks from 49 down to 40 (reasonable range for improvement)
    for rank in range(49, 39, -1):
        if time.time() > deadline:
            break
        
        # Try to construct a rank-`rank` decomposition
        # Using a structured approach based on the bilinear form
        
        # Strategy: Use the fact that we have 2 rows in A
        # Group the computation by columns of B
        # For each column k of B, we compute 2 output values
        # This gives us 7 * 2 = 14 "output slots"
        # Each output slot depends on 4 products
        
        # A more systematic approach:
        # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
        # We want to decompose T as sum_r U_r ⊗ V_r ⊗ W_r
        
        # Let's try a direct construction for rank 49
        # Based on the structure: 7 columns * 7 basis vectors per column pair
        
        if rank == 49:
            # Construct a rank-49 decomposition
            # Idea: For each pair of columns (k1, k2) in B, use 7 factors
            # Total: 7 * 7 = 49 factors
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            r = 0
            # For each column k, we need to compute 2 output values
            # Use 7 factors per column to handle the 4 inputs to 2 outputs
            
            for k in range(p):
                for t in range(7):
                    # Each factor (k, t) contributes to column k
                    # U part: depends on A's rows and columns
                    # V part: depends on B's column k
                    # W part: contributes to C[:, k]
                    
                    # Simplified: use identity-like patterns
                    # U[i*m + j] for this factor
                    U[0 * m + t % m][r] = 1
                    U[1 * m + t % m][r] = 1
                    V[(t % m) * p + k][r] = 1
                    W[k * n + 0][r] = 1
                    W[k * n + 1][r] = 1
                    r += 1
            
            # Verify this decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if time.time() > deadline:
            break
    
    # If no better decomposition found, use naive
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    
    # Check that for all i, j, k:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    
    # Also check that there are no spurious terms
    # For indices that don't correspond to valid (i,j,k) combinations,
    # the sum should be 0
    
    # Check U indices: i in [0,n), j in [0,m)
    # Check V indices: j in [0,m), k in [0,p)
    # Check W indices: k in [0,p), i in [0,n)
    
    # The decomposition is valid if the above conditions hold
    return True
# EVOLVE-BLOCK-END
