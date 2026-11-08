import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14

    # Strategy: 
    # 1. Use Strassen 2x2x2 for the 6 blocks of 2x2 in A and B corresponding to k in {0..5}.
    #    There are 2 blocks in j (0-1, 2-3) and 3 blocks in k (0-1, 2-3, 4-5).
    #    Total 6 blocks, each rank 7. Total rank 42.
    # 2. Handle the remaining k=6 terms with schoolbook.
    #    This involves A[:, j] * B[j, 6] for j in 0..3, contributing to C[:, 6].
    #    This is a 2x4 x 4x1 multiplication, rank 8.
    # Total rank 50.

    rank = 50
    U = [[0] * rank for _ in range(nA)]
    V = [[0] * rank for _ in range(nB)]
    W = [[0] * rank for _ in range(nW)]
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # A indices
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W indices (k*n + i)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00][col] += 1
        col += 1

    # Apply Strassen to the 6 blocks
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen(0, j_start, k_start)
            
    # Handle k=6 with schoolbook
    # C[:, 6] = A[:, :] * B[:, 6]
    # C[i, 6] = sum_j A[i, j] B[j, 6]
    for j in range(m):
        for i in range(n):
            u_idx = i * m + j
            v_idx = j * p + 6
            w_idx = 6 * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
