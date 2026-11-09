"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _validate_decomposition(U, V, W, n, m, p, rank):
    """Check if U, V, W form a valid decomposition of matrix multiplication."""
    # For each output element (i, l) where i in [0,n), l in [0,p):
    # sum_j A[i,j] * B[j,l] = sum_r U[i*m+j_idx] * V[j_idx*p+l] * W[l*n+i]
    # More precisely: T[i,j,l] = sum_r U[i*m + j, r] * V[j*p + l, r] * W[l*n + i, r]
    # This must equal 1 if j is the "middle" index, else 0
    # Actually the tensor is: T[i,j,k] = 1 if the multiplication uses A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all (i,j,k) in the multiplication pattern
    
    # The matrix multiplication tensor M has M[i,j,k] = 1 for all i,j,k
    # (since every input element pair contributes to some output)
    # Wait, no. The tensor T satisfies:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: sum_{i,j,k} A[i,j] * B[j,k] * C[k,i] = sum_{i,k} (A*B)[i,k] * C[k,i]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if total != 1:
                    return False
    return True

def _try_random_search(n, m, p, target_rank, max_iters, deadline):
    """Try to find a decomposition of given rank using random initialization."""
    start = time.time()
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Random initialization with small rational values
        U = [[Fraction(0)] * target_rank for _ in range(n * m)]
        V = [[Fraction(0)] * target_rank for _ in range(m * p)]
        W = [[Fraction(0)] * target_rank for _ in range(p * n)]
        
        # Initialize with random ±1 or 0
        for i in range(n * m):
            for r in range(target_rank):
                if np.random.rand() < 0.3:
                    U[i][r] = Fraction(np.random.choice([-1, 1]))
        for j in range(m * p):
            for r in range(target_rank):
                if np.random.rand() < 0.3:
                    V[j][r] = Fraction(np.random.choice([-1, 1]))
        for k in range(p * n):
            for r in range(target_rank):
                if np.random.rand() < 0.3:
                    W[k][r] = Fraction(np.random.choice([-1, 1]))
        
        # Quick check
        if _validate_decomposition(U, V, W, n, m, p, target_rank):
            return U, V, W
    return None

def _structured_decomposition(n, m, p):
    """
    Build a structured decomposition that exploits the matrix structure.
    For 2x3x7, we can potentially share computations across the 7 columns.
    
    Key insight: For fixed i and j, the term A[i,j] * B[j,k] for all k
    can be computed with shared structure. We try to find linear combinations
    that reduce the number of independent products needed.
    """
    # Start with schoolbook and try to merge terms
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
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 105  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition as fallback
    U_school, V_school, W_school, rank_school = _structured_decomposition(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Based on theoretical bounds, rank(2,3,7) is expected to be around 30-40
    # Try ranks from low to high
    
    for target_rank in range(20, rank_school + 1):
        if time.time() > deadline:
            break
        
        # Try random search for this rank
        max_iters = 50
        result = _try_random_search(n, m, p, target_rank, max_iters, deadline)
        if result is not None:
            U, V, W = result
            # Double-check validity
            if _validate_decomposition(U, V, W, n, m, p, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found the lowest rank we can
    
    return best_result
# EVOLVE-BLOCK-END
