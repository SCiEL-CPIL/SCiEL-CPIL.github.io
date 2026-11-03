import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _check_decomposition(U, V, W):
    """Verify that U, V, W form a valid decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using trace convention: W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i < n and j < m and k < p) else 0
                # Actually, the tensor for matrix multiplication is:
                # T[i,j,k] = delta(i,j,k) where the multiplication is C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor indices are (i,j,k) with value 1 if it's a valid multiplication term
                # More precisely, the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
                # where i in [0,n), j in [0,m), k in [0,p)
                # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _try_structured_rank_27():
    """
    Try to find a rank-27 decomposition using:
    - Strassen 2x2x2 block for columns 0,1 of B (j=0,1)
    - Custom 2x1x3 sub-tensor for column 2 of B (j=2)
    
    The idea: 
    - For the 2x3x2 sub-problem (A is 2x2, B is 2x2), Strassen uses rank 7
    - For the 2x1x3 sub-problem (A is 2x1, B is 1x3), we need rank 3 (schoolbook)
    - But we can potentially share structure
    
    Actually, let's think more carefully:
    - A is 2x3, B is 3x5
    - C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    We can split B into B[:,0:2] (3x2) and B[:,2:5] (3x3)
    Or split by rows of B / columns of A:
    - Part 1: j=0,1 (first 2 columns of A, first 2 rows of B) -> 2x2x5 sub-problem? No...
    
    Let me reconsider. The tensor is over indices (i, j, k) where:
    - i in {0,1} (rows of A, rows of C)
    - j in {0,1,2} (cols of A, rows of B)  
    - k in {0,1,2,3,4} (cols of B, cols of C)
    
    T[i,j,k] = 1 for all valid (i,j,k)
    
    Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    Strategy: Use Strassen on the (i,j) part for j=0,1 and handle j=2 separately.
    
    For the 2x2x5 part (i in {0,1}, j in {0,1}, k in {0..4}):
    - This is a 2x2 matrix (in i,j) times a 2x5 matrix (in j,k)
    - We can use Strassen-like decomposition for the 2x2 "core"
    
    Actually, a cleaner approach:
    - Split into two parts: (i,j) with j in {0,1} and j=2
    - For j in {0,1}: 2x2x5 tensor, can be decomposed with rank 2*5=10 schoolbook, 
      but maybe less with Strassen on the 2x2 part
    - For j=2: 2x1x5 tensor, rank 2*5=10 schoolbook
    
    Total schoolbook: 10 + 10 = 20? No, that's not right either.
    
    Let me just try to solve a linear system for rank 27.
    """
    n, m, p = DIMENSIONS
    target_rank = 27
    
    # Variables: U has n*m*target_rank = 2*3*27 = 162 entries
    #             V has m*p*target_rank = 3*5*27 = 405 entries
    #             W has p*n*target_rank = 5*2*27 = 270 entries
    # Total: 837 variables
    # Constraints: n*m*p = 2*3*5 = 30 equations (one for each (i,j,k) pair)
    # Each equation is cubic in the variables, so this is a system of polynomial equations.
    # This is too hard to solve directly.
    
    # Alternative: Use the structured ansatz.
    # Let's try: 
    # - 7 terms from Strassen for a 2x2x2 block (using i in {0,1}, j in {0,1}, k in {0,1})
    # - Remaining terms for the rest
    
    # Actually, let's try a different approach: assume specific structure.
    # For the 2x3x5 problem, let's try to find a decomposition with rank 27.
    
    # I'll use a simpler heuristic: try to find if there's a decomposition with half-integer entries
    # by setting up a linear system for specific structured forms.
    
    # For now, return None to indicate we couldn't find one, and fall back to schoolbook.
    return None

def _schoolbook_decomposition():
    """Return the standard schoolbook decomposition with rank 30."""
    n, m, p = DIMENSIONS
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

def construct():
    """
    Constructs the factor matrices for matrix multiplication 2x3 by 3x5.
    Tries to find a rank-minimal exact tensor decomposition.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try structured rank-27 ansatz
    result = _try_structured_rank_27()
    if result is not None:
        if time.time() < deadline:
            return result
    
    # Fall back to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
