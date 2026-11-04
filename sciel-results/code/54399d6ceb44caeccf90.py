"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 7]


def _build_rank1_factors(n, m, p, i, j, k):
    """Build U, V, W factor vectors for a single rank-1 term in schoolbook decomposition."""
    U = np.zeros(n * m)
    V = np.zeros(m * p)
    W = np.zeros(p * n)
    U[i * m + j] = 1
    V[j * p + k] = 1
    W[k * n + i] = 1
    return U, V, W


def _build_strassen_block_terms(n, m, p, bi, bj, bk):
    """
    Build the 7 rank-1 factor triples for a single 2x2x2 Strassen block.
    bi, bj, bk: block indices (0 or 1)
    Returns a list of 7 tuples (U, V, W)
    """
    I = bi * 2
    J = bj * 2
    K = bk * 2
    
    terms = []
    
    # M0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U0 = np.zeros(n * m); U0[I * m + J] = 1; U0[(I+1) * m + (J+1)] = 1
    V0 = np.zeros(m * p); V0[J * p + K] = 1; V0[(J+1) * p + (K+1)] = 1
    W0 = np.zeros(p * n); W0[K * n + I] = 1; W0[(K+1) * n + (I+1)] = 1
    terms.append((U0, V0, W0))
    
    # M1: (A10+A11)B00 -> C10 += 1
    U1 = np.zeros(n * m); U1[(I+1) * m + J] = 1; U1[(I+1) * m + (J+1)] = 1
    V1 = np.zeros(m * p); V1[J * p + K] = 1
    W1 = np.zeros(p * n); W1[K * n + (I+1)] = 1
    terms.append((U1, V1, W1))
    
    # M2: A00(B01-B11) -> C01 += 1, C11 += 1
    U2 = np.zeros(n * m); U2[I * m + J] = 1
    V2 = np.zeros(m * p); V2[J * p + (K+1)] = 1; V2[(J+1) * p + (K+1)] = -1
    W2 = np.zeros(p * n); W2[(K+1) * n + I] = 1; W2[(K+1) * n + (I+1)] = 1
    terms.append((U2, V2, W2))
    
    # M3: A11(B10-B00) -> C00 += 1, C10 += 1
    U3 = np.zeros(n * m); U3[(I+1) * m + (J+1)] = 1
    V3 = np.zeros(m * p); V3[(J+1) * p + K] = 1; V3[J * p + K] = -1
    W3 = np.zeros(p * n); W3[K * n + I] = 1; W3[K * n + (I+1)] = 1
    terms.append((U3, V3, W3))
    
    # M4: (A00+A01)B11 -> C00 -= 1, C01 += 1
    U4 = np.zeros(n * m); U4[I * m + J] = 1; U4[I * m + (J+1)] = 1
    V4 = np.zeros(m * p); V4[(J+1) * p + (K+1)] = 1
    W4 = np.zeros(p * n); W4[K * n + I] = -1; W4[(K+1) * n + I] = 1
    terms.append((U4, V4, W4))
    
    # M5: (A10-A00)(B00+B01) -> C11 += 1
    U5 = np.zeros(n * m); U5[(I+1) * m + J] = 1; U5[I * m + J] = -1
    V5 = np.zeros(m * p); V5[J * p + K] = 1; V5[J * p + (K+1)] = 1
    W5 = np.zeros(p * n); W5[(K+1) * n + (I+1)] = 1
    terms.append((U5, V5, W5))
    
    # M6: (A01-A11)(B10+B11) -> C00 += 1
    U6 = np.zeros(n * m); U6[I * m + (J+1)] = 1; U6[(I+1) * m + (J+1)] = -1
    V6 = np.zeros(m * p); V6[(J+1) * p + K] = 1; V6[(J+1) * p + (K+1)] = 1
    W6 = np.zeros(p * n); W6[K * n + I] = 1
    terms.append((U6, V6, W6))
    
    return terms


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: Strassen decomposition for the 4x4x4 block (columns 0-3 of B)
    # There are 2x2x2 = 8 blocks of size 2x2x2
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                terms = _build_strassen_block_terms(n, m, p, bi, bj, bk)
                for U, V, W in terms:
                    U_list.append(U)
                    V_list.append(V)
                    W_list.append(W)
    
    # Part 2: Schoolbook decomposition for the 4x4x3 block (columns 4-6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U, V, W = _build_rank1_factors(n, m, p, i, j, k)
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
    
    rank = len(U_list)
    
    # Convert to matrices
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    # Verification
    elapsed = time.time() - start_time
    if elapsed < 100:
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(s - 1) > 1e-9:
                        max_err = max(max_err, abs(s - 1))
        
        if max_err > 1e-9:
            # Fallback to schoolbook if verification fails
            U_list2 = []
            V_list2 = []
            W_list2 = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U, V, W = _build_rank1_factors(n, m, p, i, j, k)
                        U_list2.append(U)
                        V_list2.append(V)
                        W_list2.append(W)
            U_mat = np.array(U_list2).T
            V_mat = np.array(V_list2).T
            W_mat = np.array(W_list2).T
            rank = len(U_list2)

    # Convert to lists of integers
    U_out = [[int(round(U_mat[i, r])) for r in range(rank)] for i in range(n * m)]
    V_out = [[int(round(V_mat[i, r])) for r in range(rank)] for i in range(m * p)]
    W_out = [[int(round(W_mat[i, r])) for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
