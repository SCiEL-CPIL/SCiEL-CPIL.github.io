"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction as Frac

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Use a block-based construction that partitions the 7 output columns
    into overlapping 3-column blocks and exploits linear dependencies to reduce rank.
    We try to find a decomposition with rank less than 63 (schoolbook).
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # If time is running out, return schoolbook immediately
    if _time_left() < 10:
        return _schoolbook(n, m, p)
    
    # Try to find an improved decomposition
    result = _try_improved_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    
    The condition is: for all i,k,i',j',j'',k'':
    sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k'', r] = δ_{i,i'} δ_{k,k'} δ_{j',j''}
    """
    rank = len(U[0])
    
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            expected = 1.0 if (i == i2 and k == k2 and j == j2) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += W[k * n + i][r] * U[i2 * m + j][r] * V[j2 * p + k2][r]
                            if abs(actual - expected) > 1e-10:
                                return False
    return True

def _try_improved_decomposition(n, m, p):
    """
    Try to construct an improved decomposition for 3x3 * 3x7.
    
    Approach: Partition the 7 output columns into blocks and use overlapping
    3-column blocks with shared structure to reduce rank.
    
    We'll try a construction where we use 3 blocks of 3 columns each, with
    overlaps, and see if we can exploit linear dependencies.
    """
    if _time_left() < 15:
        return None
    
    # Try a rank-27 construction: 3 blocks of 3 columns each, each block
    # uses rank 9 (schoolbook for 3x3x3), but with sharing between blocks
    # to reduce total rank.
    
    # Actually, let's try a different approach: use the fact that for rectangular
    # matrix multiplication, we can sometimes do better than the naive product.
    
    # For 3x3 * 3x7, the schoolbook rank is 63. Let's try to find a lower rank
    # by using a structured decomposition.
    
    # Try rank 27: 3 blocks of 3 columns, each using rank 9
    # But this doesn't reduce rank unless we share terms.
    
    # Let's try a greedy approach: start with schoolbook and try to remove terms
    # that are linearly dependent on others.
    
    result = _schoolbook(n, m, p)
    U = [row[:] for row in result["U"]]
    V = [row[:] for row in result["V"]]
    W = [row[:] for row in result["W"]]
    rank = len(U[0])
    
    # Try to remove terms greedily
    removed = []
    for r in range(rank):
        if _time_left() < 5:
            break
        
        # Try removing term r
        U_test = [row[:r] + row[r+1:] for row in U]
        V_test = [row[:r] + row[r+1:] for row in V]
        W_test = [row[:r] + row[r+1:] for row in W]
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            # Remove this term
            U = U_test
            V = V_test
            W = W_test
            removed.append(r)
            rank -= 1
    
    if removed:
        return {"U": U, "V": V, "W": W}
    
    # Try a different approach: use a structured decomposition
    # For 3x3 * 3x7, try to use rank 21 + 6 = 27 or similar
    
    # Actually, let's try to use a known result: the rank of 3x3 * 3x7
    # might be achievable with a specific construction.
    
    # Try rank 24: use 2 blocks of 3 columns (rank 9 each = 18) plus
    # 6 additional terms for the remaining column with sharing
    result = _try_rank_24(n, m, p)
    if result is not None:
        return result
    
    # Try rank 27
    result = _try_rank_27(n, m, p)
    if result is not None:
        return result
    
    return None

def _try_rank_24(n, m, p):
    """
    Try to construct a rank-24 decomposition.
    
    Partition the 7 output columns into:
    - Block 1: columns 0, 1, 2 (3 columns)
    - Block 2: columns 3, 4, 5 (3 columns)  
    - Block 3: column 6 (1 column)
    
    Use rank 9 for each 3-column block (total 18) and 6 additional terms
    for the single column with sharing.
    """
    if _time_left() < 10:
        return None
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: columns 0, 1, 2 (terms 0-8)
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k = 0, 1, 2
                col = i * m * 3 + j * 3 + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    # Block 2: columns 3, 4, 5 (terms 9-17)
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k = 0, 1, 2 (actual col = 3+k)
                col = 9 + i * m * 3 + j * 3 + k
                U[i * m + j][col] = 1
                V[j * p + 3 + k][col] = 1
                W[(3 + k) * n + i][col] = 1
    
    # Block 3: column 6 (terms 18-23, 6 terms)
    # For a single column, we need rank 3 normally, but let's try 6 terms
    # to allow for sharing with other blocks
    for i in range(n):
        for j in range(m):
            col = 18 + i * m + j
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return result
    
    return None

def _try_rank_27(n, m, p):
    """
    Try to construct a rank-27 decomposition.
    
    Partition the 7 output columns into three blocks of 3 columns each,
    with the last block using columns 4, 5, 6.
    """
    if _time_left() < 10:
        return None
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: columns 0, 1, 2 (terms 0-8)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                col = i * m * 3 + j * 3 + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    # Block 2: columns 3, 4, 5 (terms 9-17)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                col = 9 + i * m * 3 + j * 3 + k
                U[i * m + j][col] = 1
                V[j * p + 3 + k][col] = 1
                W[(3 + k) * n + i][col] = 1
    
    # Block 3: columns 4, 5, 6 (terms 18-26) - overlapping with block 2
    # This allows for sharing
    for i in range(n):
        for j in range(m):
            for k in range(3):  # actual col = 4+k
                col = 18 + i * m * 3 + j * 3 + k
                U[i * m + j][col] = 1
                V[j * p + 4 + k][col] = 1
                W[(4 + k) * n + i][col] = 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return result
    
    return None
# EVOLVE-BLOCK-END
