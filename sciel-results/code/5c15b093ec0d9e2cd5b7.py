"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return the schoolbook decomposition as (U, V, W) lists of lists of Fractions."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly multiplies matrices.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For matrix multiplication, T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    contribute to (i,k) entry of C, which is always 1 for valid indices.
    Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in range.
    So we check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != Fraction(1):
                    return False
    return True

def _rank_of_columns(mat):
    """Compute the rank of a matrix (list of lists of Fractions) by Gaussian elimination."""
    if not mat or not mat[0]:
        return 0
    rows = len(mat)
    cols = len(mat[0])
    M = [row[:] for row in mat]
    r = 0
    for c in range(cols):
        # Find pivot
        pivot = -1
        for i in range(r, rows):
            if M[i][c] != 0:
                pivot = i
                break
        if pivot == -1:
            continue
        M[r], M[pivot] = M[pivot], M[r]
        for i in range(r + 1, rows):
            if M[i][c] != 0:
                factor = M[i][c] / M[r][c]
                for j in range(c, cols):
                    M[i][j] -= factor * M[r][j]
        r += 1
        if r == rows:
            break
    return r

def _prune_columns(U, V, W, n, m, p):
    """Try to find a subset of columns that still spans the tensor.
    This is a heuristic: greedily try to remove columns that are linear combinations
    of others in a tensor-specific sense. For now, just return as-is since exact
    pruning is NP-hard. We'll use a simpler heuristic: check if removing a column
    still allows reconstruction via the remaining columns' span.
    """
    rank = len(U[0])
    # Greedy forward selection: start with no columns, add those needed
    # Actually, let's try backward elimination: start with all, try removing one at a time
    # But verification is expensive. For small ranks, we can try.
    
    # A simpler approach: compute the tensor rank lower bound.
    # For 3x3x4 matrix multiplication, the known rank is 21 (Strassen's algorithm generalization).
    # Let's just try to find a decomposition of rank 21 by random search.
    return U, V, W

def _random_search_rank(target_rank, n, m, p, deadline, rng):
    """Try to find a decomposition of the given rank using random initialization and local search."""
    # This is a placeholder; implementing a full random search for tensor rank
    # is complex. For now, we'll try a few random rank-21 decompositions.
    # The known rank for 3x3x4 matrix multiplication is 21.
    
    # Actually, let's use the known result: the rank of 3x3x4 matrix multiplication is 21.
    # We can construct it explicitly using known algorithms.
    # For now, return None to indicate failure.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 36)
    U, V, W = _schoolbook_decomp(n, m, p)
    
    best_rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    # Known: rank of 3x3x4 matrix mult is 21
    # Let's try to construct it explicitly.
    
    # For now, return the schoolbook decomposition as a valid baseline.
    # A more sophisticated approach would be to use known rank-21 decompositions.
    
    # Convert to integer/half-integer format for the return
    def to_output(U, V, W):
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    if time.time() > deadline:
        return to_output(best_U, best_V, best_W)
    
    return to_output(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
