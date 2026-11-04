import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 5]

def _to_fraction(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, float)):
        return Fraction(x)
    return Fraction(x)

def _is_half_integer(frac):
    # Check if frac is a half-integer (i.e., frac * 2 is an integer)
    return (frac * 2).denominator == 1

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor T where T[i,j,k] = 1 for all i,j,k
    """
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2(n, m, p, j_start, k_start):
    """
    Apply Strassen-like decomposition to a 2x2 block.
    Returns list of (U_idx, V_idx, W_idx, coeffs) tuples.
    """
    # For a 2x2 block with rows i in [0,1], cols j in [j_start, j_start+1], 
    # and output cols k in [k_start, k_start+1]
    # Actually, for matrix multiplication, we need to be more careful.
    
    # The standard Strassen for 2x2 matrix multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # But here we're multiplying a 2x2 block of A (rows 0,1; cols j_start,j_start+1)
    # by a 2x2 block of B (rows j_start,j_start+1; cols k_start,k_start+1)
    # to get a 2x2 block of C (rows 0,1; cols k_start,k_start+1)
    
    terms = []
    
    # M1 = (A[0,j_start]+A[1,j_start+1])(B[j_start,k_start]+B[j_start+1,k_start+1])
    # This contributes to C[0,k_start] and C[1,k_start+1]
    terms.append([
        (0*m + j_start, 1), (1*m + j_start+1, 1),
        (j_start*p + k_start, 1), (j_start+1*p + k_start+1, 1),
        (k_start*n + 0, 1), (k_start+1*n + 1, 1)
    ])
    
    # M2 = (A[1,j_start]+A[1,j_start+1])B[j_start,k_start]
    # Contributes to C[1,k_start]
    terms.append([
        (1*m + j_start, 1), (1*m + j_start+1, 1),
        (j_start*p + k_start, 1),
        (k_start*n + 1, 1)
    ])
    
    # M3 = A[0,j_start](B[j_start+1,k_start+1]-B[j_start,k_start+1])
    # Contributes to C[0,k_start+1]
    terms.append([
        (0*m + j_start, 1),
        (j_start+1*p + k_start+1, 1), (j_start*p + k_start+1, -1),
        (k_start+1*n + 0, 1)
    ])
    
    # M4 = A[1,j_start+1](B[j_start,k_start]-B[j_start+1,k_start])
    # Contributes to C[1,k_start]
    terms.append([
        (1*m + j_start+1, 1),
        (j_start*p + k_start, 1), (j_start+1*p + k_start, -1),
        (k_start*n + 1, 1)
    ])
    
    # M5 = (A[0,j_start]+A[0,j_start+1])B[j_start+1,k_start+1]
    # Contributes to C[0,k_start+1]
    terms.append([
        (0*m + j_start, 1), (0*m + j_start+1, 1),
        (j_start+1*p + k_start+1, 1),
        (k_start+1*n + 0, 1)
    ])
    
    # M6 = (A[1,j_start]-A[0,j_start])(B[j_start,k_start+1]+B[j_start+1,k_start+1])
    # Contributes to C[1,k_start+1]
    terms.append([
        (1*m + j_start, 1), (0*m + j_start, -1),
        (j_start*p + k_start+1, 1), (j_start+1*p + k_start+1, 1),
        (k_start+1*n + 1, 1)
    ])
    
    # M7 = (A[0,j_start+1]-A[1,j_start+1])(B[j_start+1,k_start]+B[j_start+1,k_start+1])
    # Contributes to C[0,k_start]
    terms.append([
        (0*m + j_start+1, 1), (1*m + j_start+1, -1),
        (j_start+1*p + k_start, 1), (j_start+1*p + k_start+1, 1),
        (k_start*n + 0, 1)
    ])
    
    return terms

def _try_partition(n, m, p, j_groups, k_groups):
    """
    Try a partition of j dimension into j_groups and k dimension into k_groups.
    For each pair (j_group, k_group), if both have size 2, use Strassen (rank 7).
    Otherwise, use schoolbook.
    Returns the decomposition or None if invalid.
    """
    U = {}
    V = {}
    W = {}
    
    rank = 0
    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                # Use Strassen
                terms = _strassen_2x2(n, m, p, jg[0], kg[0])
                for term in terms:
                    U_idx = term[0]
                    V_idx = term[1]
                    W_idx = term[2]
                    # This is a simplified representation; need to actually build the matrices
                    pass
            else:
                # Schoolbook
                for i in range(n):
                    for j in jg:
                        for k in kg:
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            # Add rank-1 term
                            pass
    
    # This is getting complicated. Let me just try to verify if rank 27 is possible
    # by attempting to merge terms from the current rank-28 solution.
    return None

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
