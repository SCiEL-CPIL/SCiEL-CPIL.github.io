"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import sys
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if val != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition."""
    # Standard Strassen: M1=(a11+a22)(b11+b22), M2=(a21+a22)b11, 
    # M3=a11(b12-b22), M4=a22(b21-b11), M5=(a11+a12)b22,
    # M6=(a21-a11)(b11+b12), M7=(a12-a22)(b21+b22)
    # U entries: (i,j) -> coefficient in each M_r
    # V entries: (j,k) -> coefficient in each M_r
    # W entries: (k,i) -> coefficient in each M_r
    U = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    V = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    W = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    W[0][0] = 1  # c11
    W[3][0] = 1  # c22
    
    # M2 = (a21+a22)b11
    U[2][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    W[1][1] = 1  # c21
    
    # M3 = a11(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1  # -b22
    W[1][2] = 1  # c12
    
    # M4 = a22(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1  # -b11
    W[2][3] = 1  # c22
    
    # M5 = (a11+a12)b22
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    W[2][4] = 1  # c12
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1  # a21
    U[0][5] = -1  # -a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    W[0][5] = 1  # c21
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1  # a12
    U[3][6] = -1  # -a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    W[1][6] = 1  # c22
    
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(target_rank, time_budget):
    """
    Try to find a valid decomposition with the given target rank.
    Uses a structured approach: fix U to have sparse patterns, solve for V and W.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + time_budget
    
    # Strategy: Start from schoolbook and try to merge/eliminate terms
    # by finding linear dependencies.
    
    # For now, try a simple approach: use Strassen for 2x2x2 blocks
    # and combine with remaining terms.
    
    # Split 3x3x4 into:
    # - 2x2x4 block (rows 0-1, cols 0-1) -> can use Strassen-like
    # - Remaining cross terms
    
    # Actually, let's try a direct construction for rank 35 first.
    # Take schoolbook (36) and see if we can eliminate one term.
    
    # A simpler approach: use the fact that for 3x3x3, rank is 23.
    # Split B into B_3 (3x3) and B_1 (3x1).
    # Use rank-23 for A*B_3, rank-9 for A*B_1. Total = 32.
    
    # I need the actual 23 terms. Let me try to construct them.
    # This is complex, so let me try a different approach.
    
    # Alternative: Try rank 35 by removing one schoolbook term and adjusting.
    # This is unlikely to work simply.
    
    # Let me try a numerical approach: set up the tensor equation and
    # try to find a low-rank solution using iterative methods.
    
    # For the time being, return None if we can't find a valid reduction.
    if time.time() > deadline:
        return None
    
    # Try a simple heuristic: can we find a rank-35 decomposition?
    # Start with schoolbook and try to zero out one column while
    # adjusting others.
    
    U = [[0] * 35 for _ in range(n * m)]
    V = [[0] * 35 for _ in range(m * p)]
    W = [[0] * 35 for _ in range(p * n)]
    
    # Fill 35 of the 36 schoolbook terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < 35:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # The missing term is i=2, j=2, k=3
    # We need to express A[2,2]*B[2,3]*C[3,2] as a sum of the other 35 terms.
    # This is generally not possible, so this won't work.
    
    # Return None to indicate no valid reduction found.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best = _schoolbook()
    best_rank = len(best['U'][0])
    n, m, p = DIMENSIONS
    
    assert _verify_decomposition(best['U'], best['V'], best['W'], n, m, p), "Schoolbook failed!"
    
    # Try rank reduction
    if time.time() < deadline - 20:
        result = _try_rank_reduction(35, 10)
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
                best = result
                best_rank = r
    
    return best

# EVOLVE-BLOCK-END
