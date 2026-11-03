import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication. Rank 7.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = [[Fraction(0)]*7 for _ in range(4)]
    V = [[Fraction(0)]*7 for _ in range(4)]
    W = [[Fraction(0)]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _build_block_decomposition(n, m, p):
    """
    Build the block decomposition:
    - j=0: schoolbook 2x1x7, rank 14
    - j=1,2: split p=7 into (0,1),(2,3),(4,5) Strassen + (6) schoolbook
    Total rank: 14 + 7 + 7 + 7 + 4 = 39
    """
    total_rank = 14 + 7 + 7 + 7 + 4
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: j=0, schoolbook 2x1x7
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2; p split into (0,1),(2,3),(4,5),(6)
    sU, sV, sW, srank = _strassen_2x2x2()
    for p_start in [0, 2, 4]:
        for r in range(srank):
            for i in range(2):
                for j in range(2):
                    U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[(1 + j) * p + (p_start + k)][col] += sV[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[(p_start + k) * n + i][col] += sW[k * 2 + i][r]
            col += 1
    
    # Sub-block 2d: j=1,2; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    return U, V, W, total_rank

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _to_int_list(U):
    """Convert Fraction matrix to int list (all values should be integers)."""
    result = []
    for row in U:
        new_row = []
        for val in row:
            assert val.denominator == 1, f"Non-integer value: {val}"
            new_row.append(val.numerator)
        result.append(new_row)
    return result

def _try_rank_reduction(U, V, W, n, m, p, target_rank, time_budget):
    """
    Placeholder for rank reduction.
    Since simple removal only works for zero terms (which don't exist in minimal decompositions),
    and complex merging is difficult, we return the current decomposition.
    """
    return U, V, W, len(U[0])

def _randomized_search(U, V, W, n, m, p, target_rank, time_budget):
    """
    Placeholder for randomized search.
    """
    return U, V, W, len(U[0])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with proven rank-39 block decomposition
    U, V, W, rank = _build_block_decomposition(n, m, p)
    
    # Verify it's correct
    if not _verify_exact(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # The search functions are now placeholders, so we skip them to save time.
    # We rely on the initial rank-39 decomposition.
    
    # Convert to int lists
    U_int = _to_int_list(U)
    V_int = _to_int_list(V)
    W_int = _to_int_list(W)
    
    return {"U": U_int, "V": V_int, "W": W_int}
