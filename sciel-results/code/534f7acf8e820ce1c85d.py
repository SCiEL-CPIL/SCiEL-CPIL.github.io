"""Candidate for matrix multiplication tensor decomposition (3x5x5)."""
import time
import numpy as np
from itertools import product

_DIM = (3, 5, 5)

def _verify(U, V, W, n, m, p):
    """Check that the decomposition is valid."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    r = len(W[0])
    
    # For each output (i,k), the bilinear form coefficient must match
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c2 in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c2 and d == k) else 0.0
                            actual = 0.0
                            for rr in range(r):
                                actual += Wa[k*n+i, rr] * Ua[a*m+b, rr] * Va[c2*p+d, rr]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    The 7 products are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) -> coefficient
    # v_coeffs: dict mapping (j,k) -> coefficient
    # w_coeffs: dict mapping (i,k) -> coefficient
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    # M1 contributes to C11 (+1), C22 (+1)
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # M2 = (A21+A22)B11
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    # M2 contributes to C21 (+1), C22 (-1)
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # M3 = A11(B21-B22)
    u3 = {(0,0): 1}
    v3 = {(1,0): 1, (1,1): -1}
    # M3 contributes to C12 (+1), C22 (+1)
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # M4 = A22(B12-B11)
    u4 = {(1,1): 1}
    v4 = {(0,1): 1, (0,0): -1}
    # M4 contributes to C11 (+1), C21 (+1)
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # M5 = (A11+A12)B22
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    # M5 contributes to C11 (-1), C12 (+1)
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    
    # M6 = (A21-A11)(B11+B12)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    # M6 contributes to C22 (+1)
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # M7 = (A12-A22)(B21+B22)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    # M7 contributes to C11 (+1)
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def _build_decomposition(n, m, p, row_splits, col_splits, t0, deadline):
    """
    Build a decomposition using block structure with Strassen where possible.
    
    row_splits: list of row indices for A (length n)
    col_splits: list of column indices for B (length m)
    """
    # row_splits partitions the n rows of A
    # col_splits partitions the m columns of A / rows of B
    # The remaining m columns of B are handled separately
    
    # For simplicity, let's use a specific block structure:
    # Split rows of A into groups, columns into groups
    # For each block pair, use Strassen if both blocks are 2x2
    
    # Let's try a simpler approach: use Strassen on a 2x2x2 subproblem
    # and schoolbook for the rest
    
    # Strategy: 
    # - Use Strassen for the top-left 2x2x2 block (rows 0-1 of A, cols 0-1 of B, output rows 0-1)
    # - Use schoolbook for everything else
    
    # The 2x2x2 block uses rows 0,1 of A, cols 0,1 of B, and outputs to rows 0,1 of C
    # This handles C[0:2, 0:2]
    
    # Remaining terms:
    # - C[0:2, 2:5]: A[0:2, :] @ B[:, 2:5]
    # - C[2:3, 0:5]: A[2:3, :] @ B[:, 0:5]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Add Strassen terms for the 2x2x2 block (rows 0-1, cols 0-1, output rows 0-1)
    strassen_terms = _strassen_2x2x2()
    for u_dict, v_dict, w_dict in strassen_terms:
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            U_row[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            V_row[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            W_row[k * n + i] = coeff
        
        U_terms.append(U_row)
        V_terms.append(V_row)
        W_terms.append(W_row)
    
    # Add schoolbook terms for the rest
    # C[0:2, 2:5]: for i in 0,1; k in 2,3,4; j in 0,1,2,3,4
    for i in range(2):
        for j in range(m):
            for k in range(2, p):
                U_row = [0] * (n * m)
                V_row = [0] * (m * p)
                W_row = [0] * (p * n)
                U_row[i * m + j] = 1
                V_row[j * p + k] = 1
                W_row[k * n + i] = 1
                U_terms.append(U_row)
                V_terms.append(V_row)
                W_terms.append(W_row)
    
    # C[2:3, 0:5]: for i=2; k in 0,1,2,3,4; j in 0,1,2,3,4
    for i in range(2, n):
        for j in range(m):
            for k in range(p):
                U_row = [0] * (n * m)
                V_row = [0] * (m * p)
                W_row = [0] * (p * n)
                U_row[i * m + j] = 1
                V_row[j * p + k] = 1
                W_row[k * n + i] = 1
                U_terms.append(U_row)
                V_terms.append(V_row)
                W_terms.append(W_row)
    
    rank = len(U_terms)
    return {"U": U_terms, "V": V_terms, "W": W_terms}

def _try_strassen_block(n, m, p, t0, deadline):
    """
    Try to use Strassen-style block decomposition to reduce rank.
    """
    # Use Strassen for 2x2x2 subblock and schoolbook for the rest
    result = _build_decomposition(n, m, p, None, None, t0, deadline)
    return result

def _try_rank_reduction(n, m, p, t0, deadline):
    """
    Try to find a lower-rank decomposition using numerical methods.
    """
    return None

def construct():
    n, m, p = _DIM
    t0 = time.time()
    deadline = t0 + 120  # 120s safety margin (evaluator kills at 150s)
    
    # Start with schoolbook (rank 75) as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen block decomposition
    if time.time() < deadline - 30:
        result = _try_strassen_block(n, m, p, t0, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                rank = len(W[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
    
    # Try rank reduction
    if time.time() < deadline - 10:
        result = _try_rank_reduction(n, m, p, t0, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                rank = len(W[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
    
    # Verify the best result
    if _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    
    # If verification fails (shouldn't happen), return schoolbook
    return _schoolbook(n, m, p)
