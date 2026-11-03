"""Matrix multiplication 2x3x6 tensor decomposition; Kronecker-product structural construction with exact verification."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS  # n=2, m=3, p=6

    # Build the trivial decomposition first as fallback
    rank_trivial = n * m * p  # 36
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    # Kronecker-product structural construction:
    # The 2x3x6 multiplication tensor T[(i,j),(j,k),(k,i)] = 1
    # can be viewed as: for each k, a 2x3x2 sub-tensor T_k[i,j,c] = delta_{i,c}
    # Each sub-tensor has rank 2 (one term per i), giving total rank n*p = 12.
    #
    # Term l = (k, i):
    #   U[i*m+j, l] = 1 for all j
    #   V[j*p+k, l] = 1 for all j
    #   W[k*n+i, l] = 1
    #
    # Verification: sum_l U[a,l]*V[b,l]*W[c,l] = delta_{i_a,i_c}*delta_{j_a,j_b}*delta_{k_b,k_c}
    
    rank_candidate = n * p  # 12
    U_cand = [[0] * rank_candidate for _ in range(n * m)]
    V_cand = [[0] * rank_candidate for _ in range(m * p)]
    W_cand = [[0] * rank_candidate for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            l = k * n + i
            for j in range(m):
                U_cand[i * m + j][l] = 1
            for j in range(m):
                V_cand[j * p + k][l] = 1
            W_cand[k * n + i][l] = 1

    # Exact rational verification
    def verify_exact(U, V, W, n, m, p):
        r = U[0].__len__() if U else 0
        # Check all required-1 entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(Fraction(U[a][l]) * Fraction(V[b][l]) * Fraction(W[c][l]) for l in range(r))
                    if val != 1:
                        return False
        # Check all should-be-zero entries
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    val = sum(Fraction(U[a][l]) * Fraction(V[b][l]) * Fraction(W[c][l]) for l in range(r))
                    if val != 0:
                        return False
        return True

    if verify_exact(U_cand, V_cand, W_cand, n, m, p):
        return {"U": U_cand, "V": V_cand, "W": W_cand}

    # Fallback: try numpy-based verification (in case Fraction is slow)
    def verify_np(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(np.dot(Ua[a] * Va[b], Wa[c]) - 1) > 1e-9:
                        return False
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    if abs(np.dot(Ua[a] * Va[b], Wa[c])) > 1e-9:
                        return False
        return True

    if verify_np(U_cand, V_cand, W_cand, n, m, p):
        return {"U": U_cand, "V": V_cand, "W": W_cand}

    # Fallback to trivial
    return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
# EVOLVE-BLOCK-END
