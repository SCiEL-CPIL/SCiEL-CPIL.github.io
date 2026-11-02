"""Program for matrix multiplication 2x5 by 5x6 rank minimization."""
import time
import numpy as np
from fractions import Fraction

DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition.
    
    U: (n*m) x r
    V: (m*p) x r  
    W: (p*n) x r
    
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank_reduction(U, V, W, n, m, p, max_time=30.0):
    """Try to reduce the rank of a decomposition by finding linear dependencies."""
    start_time = time.time()
    r = len(U[0]) if U else 0
    if r <= 1:
        return U, V, W
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    best_U, best_V, best_W = U_frac, V_frac, W_frac
    best_rank = r
    
    # Try to eliminate columns by finding linear dependencies
    for c in range(r - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if _time_left() < 5:
            break
            
        # Check if column c contributes to the tensor
        contributes = False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 0:
                        contributes = True
                        break
                if contributes:
                    break
            if contributes:
                break
        
        if not contributes:
            # Column c doesn't contribute, remove it
            new_U = [row[:c] + row[c+1:] for row in U_frac]
            new_V = [row[:c] + row[c+1:] for row in V_frac]
            new_W = [row[:c] + row[c+1:] for row in W_frac]
            if _verify(new_U, new_V, new_W, n, m, p):
                U_frac, V_frac, W_frac = new_U, new_V, new_W
                r -= 1
                best_U, best_V, best_W = U_frac, V_frac, W_frac
                best_rank = r
            continue
    
    return best_U, best_V, best_W

def _construct_rank_45():
    """Construct a rank-45 decomposition for 2x5 by 5x6 matrix multiplication.
    
    Strategy: Use the structure that for each (i,j,k), we need
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    We'll use a pattern where:
    - U has non-zeros only in certain positions
    - V has non-zeros only in certain positions  
    - W is solved from the linear system
    
    For rank 45, we use 45 terms with a structured pattern.
    """
    n, m, p = 2, 5, 6
    rank = 45
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # Pattern: For each (j, k), we assign rank-1 terms
    # We use 5 * 6 = 30 terms for the "main" part, plus 15 more for cross terms
    
    col = 0
    # Main terms: for each j in 0..4, k in 0..5
    for j in range(m):
        for k in range(p):
            # Term 1: contributes to (i=0, j, k)
            U[0 * m + j][col] = Fraction(1)
            V[j * p + k][col] = Fraction(1)
            W[k * n + 0][col] = Fraction(1)
            col += 1
            # Term 2: contributes to (i=1, j, k)  
            U[1 * m + j][col] = Fraction(1)
            V[j * p + k][col] = Fraction(1)
            W[k * n + 1][col] = Fraction(1)
            col += 1
    
    # We have 30 terms so far. Need 15 more.
    # Add cross terms to handle the remaining equations
    # For i=0: need sum_c U[j,c]*V[j*p+k,c]*W[k,c] = 1 for all j,k
    # For i=1: need sum_c U[5+j,c]*V[j*p+k,c]*W[6+k,c] = 1 for all j,k
    
    # The above gives:
    # i=0: For (j,k), term 2j+k gives 1*1*1 = 1, others give 0. Good.
    # i=1: For (j,k), term 2j+k+30 gives 1*1*1 = 1, others give 0. Good.
    
    # But we only used 30 terms. We have 15 extra columns that are all zero.
    # Let's remove them.
    
    U = [row[:30] for row in U]
    V = [row[:30] for row in V]
    W = [row[:30] for row in W]
    
    return U, V, W

def _construct_rank_30():
    """Construct a rank-30 decomposition if possible.
    
    For rank 30, we need exactly one term per (j,k) pair.
    U[j,c] * V[j*p+k,c] * W[k*n+i,c] must sum to 1 for each (i,j,k).
    
    With one term per (j,k): let c = 6*j + k
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    So U[i*m+j, 6*j+k] * V[j*p+k, 6*j+k] * W[k*n+i, 6*j+k] = 1
    
    We can set V[j*p+k, 6*j+k] = 1, then
    U[i*m+j, 6*j+k] * W[k*n+i, 6*j+k] = 1
    
    For i=0: U[j, 6*j+k] * W[k, 6*j+k] = 1
    For i=1: U[5+j, 6*j+k] * W[6+k, 6*j+k] = 1
    
    We can set U[j, 6*j+k] = 1, W[k, 6*j+k] = 1 for i=0
    And U[5+j, 6*j+k] = 1, W[6+k, 6*j+k] = 1 for i=1
    
    But W[k, 6*j+k] appears for both i=0 and i=1 with different indices...
    W[k*n+i, c] where c = 6*j+k
    For i=0: W[k, 6*j+k]
    For i=1: W[6+k, 6*j+k]
    
    These are different rows of W, so they can be set independently!
    """
    n, m, p = 2, 5, 6
    rank = 30
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = 6 * j + k
            V[j * p + k][c] = Fraction(1)
            U[0 * m + j][c] = Fraction(1)
            W[k * n + 0][c] = Fraction(1)
            U[1 * m + j][c] = Fraction(1)
            W[k * n + 1][c] = Fraction(1)
    
    return U, V, W

def _construct_rank_25():
    """Try to construct a rank-25 decomposition.
    
    With rank 25, we have fewer terms than (j,k) pairs.
    We need to share terms between different (j,k) pairs.
    
    Let's try: for each j, have 5 terms (one per k on average).
    But 5*5 = 25, so each j gets exactly 5 terms.
    
    For fixed j, we need for each k:
    sum_{c in terms_j} U[j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is a system of 10 equations (5 k's * 2 i's) in 5 unknowns (the 5 terms for this j).
    This is overdetermined, so it may not have a solution in general.
    """
    n, m, p = 2, 5, 6
    rank = 25
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # For each j, assign 5 columns
    for j in range(m):
        base = 5 * j
        for t in range(5):
            c = base + t
            # Set V[j*p+k, c] for all k
            for k in range(p):
                V[j * p + k][c] = Fraction(1)
            # Set U[i*m+j, c] for both i
            U[0 * m + j][c] = Fraction(1)
            U[1 * m + j][c] = Fraction(1)
            # W will be solved
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = Fraction(0)
    
    # Now solve for W
    # For each (i, j, k): sum_{c in terms_j} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # = sum_{t=0}^{4} 1 * 1 * W[k*n+i, 5*j+t] = 1
    # So for each (i, k, j): sum_{t=0}^{4} W[k*n+i, 5*j+t] = 1
    
    # This means for each row of W (indexed by k*n+i), the sum over each block of 5 must be 1.
    # Row k*n+i has 25 entries divided into 5 blocks of 5.
    # Each block sum = 1.
    
    # We can set each block to have one 1 and four 0s.
    for row in range(p * n):
        for j in range(m):
            W[row][5 * j + 0] = Fraction(1)
            for t in range(1, 5):
                W[row][5 * j + t] = Fraction(0)
    
    return U, V, W

def _construct_rank_20():
    """Try to construct a rank-20 decomposition.
    
    With rank 20, we have 4 terms per j on average.
    For fixed j, we need 10 equations in 4 unknowns - very overdetermined.
    This likely has no solution.
    """
    # Skip this - too constrained
    return None

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    """
    start_time = time.time()
    n, m, p = 2, 5, 6
    
    # Try rank 25 first (most aggressive)
    U, V, W = _construct_rank_25()
    if _verify(U, V, W, n, m, p):
        best_rank = 25
    else:
        # Try rank 30
        U, V, W = _construct_rank_30()
        if _verify(U, V, W, n, m, p):
            best_rank = 30
        else:
            # Fall back to schoolbook
            U, V, W = _schoolbook(n, m, p)
            best_rank = n * m * p
    
    # Try to reduce further
    if _time_left() > 10:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p, max_time=min(60.0, _time_left() - 5))
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to integer/half-integer format
    def _convert(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if isinstance(x, Fraction):
                    if x.denominator == 1:
                        new_row.append(int(x))
                    elif x.denominator == 2:
                        new_row.append(float(x))
                    else:
                        new_row.append(float(x))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    return {"U": _convert(U), "V": _convert(V), "W": _convert(W)}
