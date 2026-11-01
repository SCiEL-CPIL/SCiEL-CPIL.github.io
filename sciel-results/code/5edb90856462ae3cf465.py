"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor using vectorized operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                
    if not np.allclose(T, target, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_block(n, m, p):
    """8 independent 2x2 Strassen blocks, rank 56."""
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        intermediates = [
            {'U': [(0,1),(3,1)], 'V': [(2*t,1),(16+2*t+1,1)], 'W': [(4*t,1),(4*t+3,1)]},
            {'U': [(2,1),(3,1)], 'V': [(2*t,1)], 'W': [(4*t+1,1),(4*t+3,-1)]},
            {'U': [(0,1)], 'V': [(2*t+1,1),(16+2*t+1,-1)], 'W': [(4*t+2,1),(4*t+3,1)]},
            {'U': [(3,1)], 'V': [(16+2*t,1),(2*t,-1)], 'W': [(4*t,1),(4*t+1,1)]},
            {'U': [(0,1),(1,1)], 'V': [(16+2*t+1,1)], 'W': [(4*t,-1),(4*t+2,1)]},
            {'U': [(2,1),(0,-1)], 'V': [(2*t,1),(2*t+1,1)], 'W': [(4*t+3,1)]},
            {'U': [(1,1),(3,-1)], 'V': [(16+2*t,1),(16+2*t+1,1)], 'W': [(4*t,1)]},
        ]
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    return U, V, W

def _strassen_paired(n, m, p):
    """
    Pair up the 8 Strassen blocks into 4 pairs. Within each pair, try to
    share the A-side linear combinations by using a combined scheme.
    
    Key idea: For two adjacent 2x2 blocks (columns 2t,2t+1 and 2t+2,2t+3),
    we can use a 2x2 x 2x4 multiplication. The rank of (2,2,4) matrix
    multiplication is known to be at most 14 (two Strassen blocks) but
    potentially less if we can share.
    
    Actually, let's try a different sharing: precompute the 7 Strassen
    A-combinations once, then for each B-block, we only need to form the
    B-side and combine. But that doesn't reduce rank in tensor form.
    
    Alternative: Use the fact that A has only 4 entries. Precompute all
    7 linear combos of A. Then for each of the 16 B-columns, we need
    products of these combos with B entries. But each product is a
    separate tensor term.
    
    Let me try: rank-4 scheme for (2,2,4) by grouping.
    For A (2x2) * B (2x4):
    We can write this as A * [B1 B2] where B1,B2 are 2x2.
    Using Strassen on each: 7+7=14 terms.
    
    But maybe we can do better. Let's try to find a rank-13 or lower
    scheme for (2,2,4) by search.
    """
    n, m, p = 2, 2, 16
    # Try a rank-14 scheme for (2,2,4) per group of 4 columns
    # This gives 4 groups * 14 = 56, same as before.
    # Let's try rank-13 for (2,2,4) if possible.
    # For now, return the standard Strassen block result.
    return _strassen_block(n, m, p)

def _search_shared_rank(n, m, p, max_rank=55, deadline=None):
    """
    Search for a decomposition with rank < 56 by exploiting the small A matrix.
    
    Strategy: Since A is only 2x2, we can precompute linear combinations.
    The tensor T[i,j,k] = sum over rank terms.
    
    Let's parameterize: U has shape (4, r), V has shape (32, r), W has shape (32, r).
    We need sum_r U[a,r]*V[b,r]*W[c,r] = delta for the right (a,b,c).
    
    For (2,2,16): a in 0..3, b in 0..31, c in 0..31.
    The nonzero entries are: for i in 0..1, j in 0..1, k in 0..15:
      a = i*2+j, b = j*16+k, c = k*2+i
    
    So we need: for each (i,j,k): sum_r U[i*2+j,r] * V[j*16+k,r] * W[k*2+i,r] = 1
    And for all other (a,b,c): the sum = 0.
    
    This is a system of polynomial equations. Hard to solve directly.
    
    Instead, let's try a structured approach: assume U has a specific form
    based on Strassen, and search for V, W with fewer columns.
    """
    if deadline is None:
        deadline = time.time() + 110
    
    start_time = time.time()
    
    # Try rank values from 55 down
    for target_rank in range(55, 31, -1):
        if time.time() > deadline:
            break
        
        # Build the constraint system
        # Unknowns: U (4 x r), V (32 x r), W (32 x r)
        # Constraints: for each (a,b,c) in the support, the trilinear form = 1
        #              for each (a,b,c) not in support, the trilinear form = 0
        
        # This is too large to solve directly. Instead, use a heuristic:
        # Start from Strassen rank-56 and try to merge columns.
        
        if target_rank >= 56:
            continue
        
        # For now, just return None to indicate no improvement found
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Try Strassen-based construction (rank 56)
    U, V, W = _strassen_block(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
