"""
Matrix multiplication 3x5x5 tensor decomposition.
Uses recursive block splitting with Strassen for 2x2x2 sub-blocks.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term is (U_dict, V_dict, W_dict) where:
      U_dict maps (i,j) -> coefficient of A[i,j]
      V_dict maps (j,k) -> coefficient of B[j,k]
      W_dict maps (k,i) -> coefficient in C[i,k]
    """
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2 = (a10+a11)b00
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    # M3 = a00(b01-b11)
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    # M4 = a11(b10-b00)
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    # M5 = (a00+a01)b11
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    # M6 = (a10-a00)(b00+b01)
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7 = (a01-a11)(b10+b11)
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms


def _schoolbook_terms(n, m, p):
    """Return schoolbook rank-1 terms for (n,m,p)."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((
                    {(i, j): 1},
                    {(j, k): 1},
                    {(k, i): 1}
                ))
    return terms


def _decompose(n, m, p, deadline=None):
    """
    Recursively decompose the (n,m,p) matrix multiplication tensor.
    Returns a list of (U_dict, V_dict, W_dict) with local indices.
    """
    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_terms()
    if n == 1 or m == 1 or p == 1:
        return _schoolbook_terms(n, m, p)
    
    # Try to find a beneficial split.
    # We split the middle dimension m into m1 + m2.
    # The two sub-problems are (n, m1, p) and (n, m2, p).
    # We want to choose m1 such that both sub-problems can potentially use Strassen.
    
    # For (n, m, p), we can split m into m1 and m2.
    # If m1 == 2 and n == 2 and p == 2, we can use Strassen for the first part.
    # Similarly for the second part.
    
    # Strategy: try to extract 2x2x2 blocks where possible.
    # For n=3, m=5, p=5:
    # Split m into 2 + 3.
    #   Sub-problem 1: (3, 2, 5) - can't use Strassen directly (n!=2 or p!=2)
    #   Sub-problem 2: (3, 3, 5) - can't use Strassen directly
    #
    # Let's think more carefully. Strassen applies when n=m=p=2.
    # We can only use Strassen if all three dimensions are 2.
    #
    # For (3,5,5), we can't directly apply Strassen.
    # But we can split and recurse.
    #
    # Let's try splitting m into 2 + 3:
    #   (3,2,5): split m into 1+1 or just schoolbook (2*3*5=30 terms)
    #   (3,3,5): split m into 1+2 or 2+1
    #     (3,1,5): schoolbook (15 terms)
    #     (3,2,5): schoolbook (30 terms) -> total 45
    #   Total: 30 + 45 = 75. Same as schoolbook.
    #
    # What if we split n instead?
    # C = A*B, A is 3x5, B is 5x5, C is 3x5.
    # Split n into 1+2:
    #   (1,5,5): schoolbook (25 terms)
    #   (2,5,5): split m into 2+3
    #     (2,2,5): split p into 2+3
    #       (2,2,2): Strassen (7 terms)
    #       (2,2,3): split p into 1+2
    #         (2,2,1): schoolbook (4 terms)
    #         (2,2,2): Strassen (7 terms)
    #       Total for (2,2,3): 4+7=11
    #     Total for (2,2,5): 7+11=18
    #     (2,3,5): split m into 1+2
    #       (2,1,5): schoolbook (10 terms)
    #       (2,2,5): 18 terms (from above)
    #     Total for (2,3,5): 10+18=28
    #   Total for (2,5,5): 18+28=46
    # Total: 25+46=71.
    
    # So the strategy is:
    # 1. If n==2 and m==2 and p==2, use Strassen.
    # 2. Otherwise, try splitting n, m, or p.
    # 3. Choose the split that minimizes total rank.
    
    best_terms = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        terms1 = _decompose(n1, m, p, deadline)
        terms2 = _decompose(n2, m, p, deadline)
        rank = len(terms1) + len(terms2)
        if rank < best_rank:
            best_rank = rank
            # Shift indices for terms2: i offset by n1
            shifted_terms2 = []
            for U_d, V_d, W_d in terms2:
                new_W_d = {}
                for (k, i), val in W_d.items():
                    new_W_d[(k, i + n1)] = val
                shifted_terms2.append((U_d, V_d, new_W_d))
            best_terms = terms1 + shifted_terms2
    
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        terms1 = _decompose(n, m1, p, deadline)
        terms2 = _decompose(n, m2, p, deadline)
        rank = len(terms1) + len(terms2)
        if rank < best_rank:
            best_rank = rank
            # Shift indices for terms2: j offset by m1
            shifted_terms2 = []
            for U_d, V_d, W_d in terms2:
                new_U_d = {}
                for (i, j), val in U_d.items():
                    new_U_d[(i, j + m1)] = val
                new_V_d = {}
                for (j, k), val in V_d.items():
                    new_V_d[(j + m1, k)] = val
                shifted_terms2.append((new_U_d, new_V_d, W_d))
            best_terms = terms1 + shifted_terms2
    
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        terms1 = _decompose(n, m, p1, deadline)
        terms2 = _decompose(n, m, p2, deadline)
        rank = len(terms1) + len(terms2)
        if rank < best_rank:
            best_rank = rank
            # Shift indices for terms2: k offset by p1
            shifted_terms2 = []
            for U_d, V_d, W_d in terms2:
                new_V_d = {}
                for (j, k), val in V_d.items():
                    new_V_d[(j, k + p1)] = val
                new_W_d = {}
                for (k, i), val in W_d.items():
                    new_W_d[(k + p1, i)] = val
                shifted_terms2.append((U_d, new_V_d, new_W_d))
            best_terms = terms1 + shifted_terms2
    
    if best_terms is not None:
        return best_terms
    else:
        return _schoolbook_terms(n, m, p)


def _build_matrices(terms, n, m, p):
    """
    Convert a list of (U_dict, V_dict, W_dict) terms into U, V, W matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_d, V_d, W_d) in enumerate(terms):
        for (i, j), val in U_d.items():
            U[i * m + j][c] = val
        for (j, k), val in V_d.items():
            V[j * p + k][c] = val
        for (k, i), val in W_d.items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Uses recursive block splitting with Strassen for 2x2x2 sub-blocks.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    terms = _decompose(n, m, p, deadline)
    
    result = _build_matrices(terms, n, m, p)
    
    return result

# EVOLVE-BLOCK-END
