"""
Tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
Uses structured random search to find a rank-minimal exact tensor decomposition.
"""
import time
import sys
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = delta_{i,k} should be reconstructed.
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _is_halfint(val, tol=1e-9):
    """Check if val is an integer or half-integer within tolerance."""
    doubled = val * 2
    nearest = round(doubled)
    return abs(doubled - nearest) < tol

def _check_halfint(U, V, W, tol=1e-9):
    """Check if all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_halfint(val, tol):
                    return False
    return True

def _random_structured(n, m, p, rank, seed=None):
    """
    Generate a random structured decomposition with the given rank.
    Uses a structured approach where we try to find sparse, low-rank components.
    """
    if seed is not None:
        random.seed(seed)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strategy: for each rank component, randomly choose which (i,j,k) triple it "targets"
    # and set the corresponding entries to 1 or -1
    for c in range(rank):
        # Randomly choose a pattern for this component
        # Each component can contribute to multiple (i,j,k) entries
        # We use a structured pattern: choose a subset of indices
        
        # For U: choose which (i,j) pairs get non-zero values
        for i in range(n):
            for j in range(m):
                if random.random() < 0.3:
                    U[i*m+j, c] = random.choice([1, -1])
        
        # For V: choose which (j,k) pairs get non-zero values
        for j in range(m):
            for k in range(p):
                if random.random() < 0.3:
                    V[j*p+k, c] = random.choice([1, -1])
        
        # For W: choose which (k,i) pairs get non-zero values
        for k in range(p):
            for i in range(n):
                if random.random() < 0.3:
                    W[k*n+i, c] = random.choice([1, -1])
    
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition with rank n*m*p."""
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
    
    return U, V, W

def construct():
    """
    Construct the tensor decomposition for 3x3x8 matrix multiplication.
    Returns {'U': (9, rank), 'V': (24, rank), 'W': (24, rank)}
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But we need to find exact decompositions with half-integer entries
    
    # Search from lower ranks upward
    min_rank = max(n*m, m*p, p*n)  # 24
    max_search_rank = best_rank - 1  # Try to beat schoolbook
    
    # Use structured random search
    for rank in range(min_rank, max_search_rank + 1):
        if time.time() > deadline - 5:  # Stop if close to deadline
            break
        
        # Try multiple random restarts for this rank
        num_restarts = 100
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            U, V, W = _random_structured(n, m, p, rank, seed=restart * 1000 + rank)
            
            # Verify correctness
            if _verify(U, V, W, n, m, p):
                # Check if entries are half-integers
                if _check_halfint(U, V, W):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    # Found a valid decomposition, try to find even lower rank
                    break  # Break out of restart loop, try lower rank
        
        # If we found a valid decomposition at this rank, we can stop
        # (since we're searching from low to high)
        if best_rank == rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
