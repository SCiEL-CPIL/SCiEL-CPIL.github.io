import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify the tensor decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    And sum_c U[i*m+j, c] * V[j2*p+k, c] * W[k2*n+i2, c] == 0 if j!=j2 or k!=k2
    """
    rank = U.shape[1]
    
    # Check diagonal (valid) entries: must be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for c_idx in range(rank):
                    val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries: must be 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if j == j2 and k == k2:
                                continue
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            val = 0.0
                            for c_idx in range(rank):
                                val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                            if abs(val) > tol:
                                return False
    return True

def _is_integer_or_half_integer(arr, tol=1e-9):
    """Check if all entries are integers or half-integers."""
    for val in arr.flatten():
        if abs(val * 2 - round(val * 2)) > tol:
            return False
    return True

def _round_to_integer_or_half_integer(arr):
    """Round entries to the nearest integer or half-integer."""
    return np.round(arr * 2) / 2.0

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are length-2 vectors.
    The decomposition satisfies: sum_c u_c[i] * v_c[j] * w_c[k] = delta_{i,j,k}
    for the 2x2x2 multiplication tensor.
    """
    # Strassen's 7 terms for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # In tensor form, each term c has vectors u_c (for A), v_c (for B), w_c (for C).
    # u_c[i,j], v_c[j,k], w_c[k,i] such that sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = delta_{i,j,k}
    
    terms = []
    
    # M1: u = [1, 0, 0, 1] (A11+A22), v = [1, 0, 0, 1] (B11+B22), w = [1, 0, 0, -1] (C11-C22)
    # Wait, let me be more careful with the index mapping.
    # For 2x2x2: i,j in {0,1}, k in {0,1}
    # u index: i*2+j -> 0,1,2,3
    # v index: j*2+k -> 0,1,2,3
    # w index: k*2+i -> 0,1,2,3
    
    # M1: (A00+A11)(B00+B11) contributes to C00 and -C11
    # u: [1, 0, 0, 1], v: [1, 0, 0, 1], w: [1, 0, 0, -1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]))
    
    # M2: (A10+A11)B00 contributes to C10 and -C11
    # u: [0, 0, 1, 1], v: [1, 0, 0, 0], w: [0, 0, 1, -1]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    
    # M3: A00(B10-B11) contributes to C01 and C11
    # u: [1, 0, 0, 0], v: [0, 0, 1, -1], w: [0, 1, 0, 1]
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]))
    
    # M4: A11(B01-B00) contributes to C00 and C10
    # u: [0, 0, 0, 1], v: [0, 1, -1, 0], w: [1, 0, 1, 0]
    terms.append(([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]))
    
    # M5: (A00+A01)B11 contributes to -C00 and C01
    # u: [1, 1, 0, 0], v: [0, 0, 0, 1], w: [-1, 1, 0, 0]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # M6: (A10-A00)(B00+B01) contributes to C11
    # u: [-1, 0, 1, 0], v: [1, 1, 0, 0], w: [0, 0, 0, 1]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # M7: (A01-A11)(B10+B11) contributes to C00
    # u: [0, 1, 0, -1], v: [0, 0, 1, 1], w: [1, 0, 0, 0]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return terms

def _embed_strassen_term(u2, v2, w2, n, m, p, i_offset=0, j_offset=0, k_offset=0):
    """
    Embed a 2x2x2 Strassen term into the 3x3x5 problem.
    u2, v2, w2 are length-4 vectors (flattened 2x2).
    Returns (u, v, w) as length n*m, m*p, p*n vectors.
    """
    u = np.zeros(n * m)
    v = np.zeros(m * p)
    w = np.zeros(p * n)
    
    # u2 indices: i2*2+j2 -> map to (i_offset+i2, j_offset+j2) in n*m
    for i2 in range(2):
        for j2 in range(2):
            idx2 = i2 * 2 + j2
            idx = (i_offset + i2) * m + (j_offset + j2)
            u[idx] = u2[idx2]
    
    # v2 indices: j2*2+k2 -> map to (j_offset+j2, k_offset+k2) in m*p
    for j2 in range(2):
        for k2 in range(2):
            idx2 = j2 * 2 + k2
            idx = (j_offset + j2) * p + (k_offset + k2)
            v[idx] = v2[idx2]
    
    # w2 indices: k2*2+i2 -> map to (k_offset+k2, i_offset+i2) in p*n
    for k2 in range(2):
        for i2 in range(2):
            idx2 = k2 * 2 + i2
            idx = (k_offset + k2) * n + (i_offset + i2)
            w[idx] = w2[idx2]
    
    return u, v, w

def _construct_with_strassen(n, m, p):
    """
    Construct a decomposition using Strassen for the 2x2x2 sub-block (i,j,k in {0,1})
    and schoolbook for all other (i,j,k) triples.
    Rank = 7 + (n*m*p - 8) = n*m*p - 1
    """
    U_list = []
    V_list = []
    W_list = []
    
    # Add Strassen terms for the 2x2x2 block at (i,j,k) in {0,1}
    strassen_terms = _strassen_2x2_terms()
    for u2, v2, w2 in strassen_terms:
        u, v, w = _embed_strassen_term(u2, v2, w2, n, m, p, i_offset=0, j_offset=0, k_offset=0)
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Add schoolbook terms for all (i,j,k) except those in {0,1}^3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # 1. Schoolbook decomposition (rank 45)
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = n * m * p
    
    # 2. Try Strassen-based construction (rank 44)
    if time.time() < deadline:
        try:
            U_str, V_str, W_str = _construct_with_strassen(n, m, p)
            rank_str = U_str.shape[1]
            if rank_str < best_rank:
                # Verify
                if _verify_decomposition(U_str, V_str, W_str, n, m, p):
                    if _is_integer_or_half_integer(U_str) and _is_integer_or_half_integer(V_str) and _is_integer_or_half_integer(W_str):
                        best_U = U_str
                        best_V = V_str
                        best_W = W_str
                        best_rank = rank_str
        except Exception:
            pass
    
    # 3. Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
