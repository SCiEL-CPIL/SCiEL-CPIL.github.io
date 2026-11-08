"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 110  # Safety margin

    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid.
        Condition: for all i in [0,n), j in [0,m), k in [0,p):
            sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def strassen_2x2x2(j0, j1, k0, k1):
        """Return 7 rank-1 terms for the (2,2,2) sub-tensor with j in {j0,j1}, k in {k0,k1}."""
        terms = []
        # M1 = (a+d)(e+h)
        terms.append(({0*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1, j1*p+k1: 1}, {k0*n+0: 1, k1*n+1: 1}))
        # M2 = (c+d)e
        terms.append(({1*m+j0: 1, 1*m+j1: 1}, {j0*p+k0: 1}, {k0*n+1: 1}))
        # M3 = a(f-h)
        terms.append(({0*m+j0: 1}, {j0*p+k1: 1, j1*p+k1: -1}, {k1*n+0: 1}))
        # M4 = d(g-e)
        terms.append(({1*m+j1: 1}, {j1*p+k0: 1, j0*p+k0: -1}, {k0*n+1: 1}))
        # M5 = (a+b)h
        terms.append(({0*m+j0: 1, 0*m+j1: 1}, {j1*p+k1: 1}, {k0*n+0: -1, k1*n+1: 1}))
        # M6 = (c+a)g
        terms.append(({1*m+j0: 1, 0*m+j0: 1}, {j1*p+k0: 1}, {k0*n+1: 1}))
        # M7 = (b-d)(e+f)
        terms.append(({0*m+j1: 1, 1*m+j1: -1}, {j0*p+k0: 1, j1*p+k0: 1}, {k0*n+0: 1}))
        return terms

    def naive_2x1x_p(j0):
        """Return 1 rank-1 term for the (2,1,p) sub-tensor with j=j0."""
        terms = []
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for i in range(n):
            u_dict[i*m+j0] = 1
        for k in range(p):
            v_dict[j0*p+k] = 1
        for k in range(p):
            for i in range(n):
                w_dict[k*n+i] = 1
        terms.append((u_dict, v_dict, w_dict))
        return terms

    def naive_2x2x2(j0, j1, k0, k1):
        """Return 4 naive rank-1 terms for (2,2,2) with j in {j0,j1}, k in {k0,k1}."""
        terms = []
        for i in range(n):
            for j in (j0, j1):
                for k in (k0, k1):
                    terms.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
        return terms

    # Kronecker decomposition: split j into {0,1} and {2,3,4}
    # Sub-problem 1: (2,2,6) with j in {0,1}
    # Sub-problem 2: (2,3,6) with j in {2,3,4}
    
    terms = []
    
    # Sub-problem 1: (2,2,6) - j in {0,1}
    # Split k into pairs: (0,1), (2,3), (4,5)
    for k0, k1 in [(0,1), (2,3), (4,5)]:
        terms.extend(strassen_2x2x2(0, 1, k0, k1))
    # 3 * 7 = 21 terms
    
    # Sub-problem 2: (2,3,6) - j in {2,3,4}
    # Split j into {2,3} and {4}
    # j in {2,3}: (2,2,6) - split k into pairs
    for k0, k1 in [(0,1), (2,3), (4,5)]:
        terms.extend(strassen_2x2x2(2, 3, k0, k1))
    # 3 * 7 = 21 terms
    
    # j = 4: (2,1,6) - rank 1
    terms.extend(naive_2x1x_p(4))
    # 1 term
    
    total_rank = len(terms)
    
    # Build matrices
    U_mat = [[0] * total_rank for _ in range(n * m)]
    V_mat = [[0] * total_rank for _ in range(m * p)]
    W_mat = [[0] * total_rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U_mat[idx][c] = val
        for idx, val in v_dict.items():
            V_mat[idx][c] = val
        for idx, val in w_dict.items():
            W_mat[idx][c] = val
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    # Fallback: naive rank-60
    rank_naive = n * m * p
    U_naive = [[0] * rank_naive for _ in range(n * m)]
    V_naive = [[0] * rank_naive for _ in range(m * p)]
    W_naive = [[0] * rank_naive for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    return {"U": U_naive, "V": V_naive, "W": W_naive}
# EVOLVE-BLOCK-END
