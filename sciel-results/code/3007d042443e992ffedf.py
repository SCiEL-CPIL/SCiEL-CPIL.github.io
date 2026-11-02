"""
Matrix multiplication 3x3 by 3x7: rank-minimal exact tensor decomposition.

Strategy:
  1. Model the tensor decomposition as a constraint satisfaction problem over F_2
     to find a valid sparsity pattern (support) for a low-rank decomposition.
  2. Lift the binary support to a system of linear equations over the rationals
     and solve for exact integer/half-integer coefficients.
  3. If the linear system is inconsistent, try alternative supports or fall back
     to the schoolbook decomposition (rank 63).

The key insight: over F_2, each rank term with support (S_U, S_V, S_W) contributes
to entries (i,j,k) where i*m+j in S_U, j*p+k in S_V, k*n+i in S_W. We search for
supports such that every (i,j,k) is covered an odd number of times, then lift to
exact rational coefficients.
"""

import time
from fractions import Fraction
import random
import sys
from itertools import product

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify exact tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return (U, V, W)

def _f2_check_support(U_support, V_support, W_support, n, m, p):
    """
    Check if the given binary supports satisfy the F_2 constraint:
    For each (i,j,k), the number of rank terms r where
      (i*m+j) in U_support[r], (j*p+k) in V_support[r], (k*n+i) in W_support[r]
    must be odd (== 1 mod 2).
    """
    rank = len(U_support)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                count = 0
                for r in range(rank):
                    if (i*m+j) in U_support[r] and (j*p+k) in V_support[r] and (k*n+i) in W_support[r]:
                        count += 1
                if count % 2 != 1:
                    return False
    return True

def _lift_to_rationals(U_support, V_support, W_support, n, m, p):
    """
    Given binary supports, try to find rational coefficients.
    For each rank term r, the coefficient c_r must satisfy:
      sum_r c_r * [i*m+j in U_support[r]] * [j*p+k in V_support[r]] * [k*n+i in W_support[r]] = 1
    
    This is a linear system in the c_r's if we fix which entries are non-zero.
    But the coefficients multiply three binary indicators, so it's still linear in c_r.
    
    Actually, we can assign a coefficient to each (entry, rank) pair. Let's use a
    simpler approach: assign coefficient 1 or -1 to each non-zero entry, and check
    if the product gives the right sum.
    
    Better: set all non-zero entries to 1 and check. If that works, done.
    If not, try flipping signs.
    """
    rank = len(U_support)
    
    # First try: all coefficients = 1
    U = [[1 if (i*m+j) in U_support[r] else 0 for r in range(rank)] for i in range(n) for j in range(m)]
    V = [[1 if (j*p+k) in V_support[r] else 0 for r in range(rank)] for j in range(m) for k in range(p)]
    W = [[1 if (k*n+i) in W_support[r] else 0 for r in range(rank)] for k in range(p) for i in range(n)]
    
    if _verify(U, V, W, n, m, p):
        return (U, V, W)
    
    # Try: solve linear system. For each (i,j,k):
    #   sum_r [i*m+j in U_sup[r]] * [j*p+k in V_sup[r]] * [k*n+i in W_sup[r]] * x_r = 1
    # where x_r is the coefficient for rank term r (assuming all entries in a term have same coeff)
    
    # Build linear system: A * x = b
    # A[(i,j,k), r] = 1 if (i*m+j in U_sup[r]) and (j*p+k in V_sup[r]) and (k*n+i in W_sup[r])
    # b[(i,j,k)] = 1
    
    import numpy as np
    A = np.zeros((n*m*p, rank), dtype=object)
    b = np.ones(n*m*p, dtype=object)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    if (i*m+j) in U_support[r] and (j*p+k) in V_support[r] and (k*n+i) in W_support[r]:
                        A[idx, r] = 1
                idx += 1
    
    # Try to solve using Gaussian elimination over rationals
    # Use Fraction for exactness
    from fractions import Fraction as Fr
    A_frac = [[Fr(A[i,j]) for j in range(rank)] for i in range(n*m*p)]
    b_frac = [Fr(1) for _ in range(n*m*p)]
    
    # Gaussian elimination
    n_eq = n*m*p
    n_var = rank
    if n_var > n_eq:
        # Underdetermined - try setting free variables to 0
        pass
    
    # Perform elimination
    M = [row[:] + [b_frac[i]] for i, row in enumerate(A_frac)]
    
    col_used = 0
    for row in range(n_eq):
        if col_used >= n_var:
            break
        # Find pivot
        pivot = -1
        for r in range(row, n_eq):
            if M[r][col_used] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        M[row], M[pivot] = M[pivot], M[row]
        # Normalize
        pivot_val = M[row][col_used]
        for j in range(col_used, n_var + 1):
            M[row][j] = M[row][j] / pivot_val
        # Eliminate
        for r in range(n_eq):
            if r != row and M[r][col_used] != 0:
                factor = M[r][col_used]
                for j in range(col_used, n_var + 1):
                    M[r][j] -= factor * M[row][j]
        col_used += 1
    
    # Extract solution
    x = [Fr(0)] * n_var
    for row in range(min(col_used, n_eq)):
        # Find which variable this row corresponds to
        for j in range(n_var):
            if M[row][j] != 0:
                x[j] = M[row][n_var]
                break
    
    # Check if solution is valid
    U = [[(x[r] if (i*m+j) in U_support[r] else Fr(0)) for r in range(rank)] for i in range(n) for j in range(m)]
    V = [[(x[r] if (j*p+k) in V_support[r] else Fr(0)) for r in range(rank)] for j in range(m) for k in range(p)]
    W = [[(x[r] if (k*n+i) in W_support[r] else Fr(0)) for r in range(rank)] for k in range(p) for i in range(n)]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Convert to regular numbers
        U_out = [[float(U[i][r]) if U[i][r] != Fr(0) else 0 for r in range(rank)] for i in range(n*m)]
        V_out = [[float(V[i][r]) if V[i][r] != Fr(0) else 0 for r in range(rank)] for i in range(m*p)]
        W_out = [[float(W[i][r]) if W[i][r] != Fr(0) else 0 for r in range(rank)] for i in range(p*n)]
        return (U_out, V_out, W_out)
    
    return None

def _try_structured_decomposition(n, m, p, time_limit=120):
    """
    Try structured decompositions based on sharing the j-summation structure.
    
    Key idea: For the inner product over j (length m=3), we can use a fixed
    set of 'basis' patterns. If we define m intermediate variables per (i,k) pair,
    we get rank n*p*m = 63 (schoolbook). But if we share across i or k, we can reduce.
    
    Alternative: Use the fact that for each fixed j, we have n*p = 21 independent
    products. The tensor can be written as sum_j of (column j of A) outer (row j of B).
    Each such term has rank n*p = 21, giving total rank 3*21 = 63.
    
    To go below 63, we need to share structure across j values.
    
    Let's try: for each (i,k), the value is sum_j A[i,j]*B[j,k].
    If we precompute, for each j, the vector a_j = (A[0,j], A[1,j], A[2,j]) and
    b_j = (B[j,0], ..., B[j,6]), then C[i,k] = sum_j a_j[i] * b_j[k].
    
    This is a sum of 3 outer products of vectors of length 3 and 7.
    The tensor rank of sum of 3 outer products is at most 3*3*7 = 63, but
    can be lower if the vectors are linearly dependent in a useful way.
    
    Actually, the rank of the sum of outer products v1*w1 + v2*w2 + v3*w3
    where v_i in R^3, w_i in R^7 is at most 3*3*7 = 63, but the bilinear rank
    (which is what we need) might be lower.
    
    Let me try a direct approach: enumerate small structured patterns.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Try rank from lower bound up
    rank_lb = max(n*m, m*p, n*p)
    
    for r in range(rank_lb, min(rank_lb + 20, n*m*p + 1)):
        if time.time() > deadline - 5:
            break
        
        # Strategy 1: Each rank term covers a "block" of (i,j,k) entries
        # Try patterns where U has a few non-zeros per column
        
        random.seed(12345 + r)
        
        for attempt in range(500):
            if time.time() > deadline - 3:
                break
            
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            for col in range(r):
                # Choose a random but structured pattern
                # Pick 1-3 i-values, 1-3 j-values, 1-3 k-values
                n_i = random.randint(1, min(2, n))
                n_j = random.randint(1, min(2, m))
                n_k = random.randint(1, min(2, p))
                
                is_ = random.sample(range(n), n_i)
                js = random.sample(range(m), n_j)
                ks = random.sample(range(p), n_k)
                
                for i in is_:
                    for j in js:
                        U[i*m+j][col] = random.choice([1, -1])
                for j in js:
                    for k in ks:
                        V[j*p+k][col] = random.choice([1, -1])
                for k in ks:
                    for i in is_:
                        W[k*n+i][col] = random.choice([1, -1])
            
            if _verify(U, V, W, n, m, p):
                return (U, V, W)
        
        # Strategy 2: F_2 support search with random restarts
        for attempt in range(200):
            if time.time() > deadline - 3:
                break
            
            U_sup = [set() for _ in range(r)]
            V_sup = [set() for _ in range(r)]
            W_sup = [set() for _ in range(r)]
            
            for col in range(r):
                # Random support sizes
                su = random.randint(1, 4)
                sv = random.randint(1, 6)
                sw = random.randint(1, 6)
                
                for idx in random.sample(range(n*m), min(su, n*m)):
                    U_sup[col].add(idx)
                for idx in random.sample(range(m*p), min(sv, m*p)):
                    V_sup[col].add(idx)
                for idx in random.sample(range(p*n), min(sw, p*n)):
                    W_sup[col].add(idx)
            
            if _f2_check_support(U_sup, V_sup, W_sup, n, m, p):
                result = _lift_to_rationals(U_sup, V_sup, W_sup, n, m, p)
                if result is not None:
                    return result
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Get schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try structured decomposition search
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        result = _try_structured_decomposition(n, m, p, time_limit=remaining - 5)
        if result is not None:
            U, V, W = result
            r = len(U[0])
            if r < best_rank:
                best_result = (U, V, W)
                best_rank = r
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
