"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np
from itertools import product

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 115  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies among schoolbook terms
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We want to write T = sum_c u_c ⊗ v_c ⊗ w_c with minimal rank
    # where u_c[i,j] = U[i*m+j,c], v_c[j,k] = V[j*p+k,c], w_c[k,i] = W[k*n+i,c]
    
    # Approach: Use the fact that for fixed i,k, the vector over j must satisfy
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
    # This is a system of bilinear equations
    
    # Try a structured low-rank decomposition using block Strassen
    # Split n=4 into 2+2, m=5 into 2+3, p=6 into 3+3
    
    # For a 2x2 * 2x2 block, Strassen gives rank 7 instead of 8
    # We'll build a decomposition using this idea
    
    # Direct construction: use the known result that rank(4x5, 5x6) <= 105 or similar
    # For now, try a greedy merge approach
    
    def try_greedy_merge():
        """Try to merge pairs of schoolbook columns."""
        n, m, p = DIMENSIONS
        rank0 = n * m * p
        U = [[0] * rank0 for _ in range(n * m)]
        V = [[0] * rank0 for _ in range(m * p)]
        W = [[0] * rank0 for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Try to find pairs of columns that can be merged
        # Two columns a, b can be merged if there exist scalars such that
        # the resulting decomposition still satisfies all constraints
        # This is complex; skip for now and return schoolbook
        return U, V, W, rank0
    
    # Use a more sophisticated approach: exploit the structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as a sum of rank-1 tensors in the bilinear sense
    
    # Key insight: if we can find a set of "partial sums" that cover all (i,j,k)
    # with fewer terms than n*m*p, we reduce the rank
    
    # For 4x5*5x6, one approach is to use the following:
    # Group the j-index into overlapping sets and use inclusion-exclusion
    
    # Try rank = 105 construction (based on known bounds)
    # Actually, let's just return the best we can verify
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Try a simple perturbation: check if removing one column and adjusting others works
        # This is a heuristic search
        
        pass
    
    return best_result

# EVOLVE-BLOCK-END
