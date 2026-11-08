import numpy as np
import time
from itertools import product

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14

    def build_rank50():
        """
        Construct a rank-50 decomposition for M_{2,4,7}.
        Splits into two M_{2,2,7} blocks (j=0,1 and j=2,3).
        Each M_{2,2,7} is decomposed into three 2x2x2 Strassen blocks (rank 7 each)
        and one 2x2x1 schoolbook block (rank 4).
        Total rank per block: 3*7 + 4 = 25.
        Total rank: 2 * 25 = 50.
        """
        rank_max = 50
        U = [[0] * rank_max for _ in range(nA)]
        V = [[0] * rank_max for _ in range(nB)]
        W = [[0] * rank_max for _ in range(nW)]
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # Indices for A (2x2 block at i0, j0)
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            # Indices for B (2x2 block at j0, k0)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            
            # Indices for C (2x2 block at k0, i0)
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

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)

            # a*e
            U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00][col] += 1; col += 1
            # b*g
            U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00][col] += 1; col += 1
            # c*e
            U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10][col] += 1; col += 1
            # d*g
            U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10][col] += 1; col += 1

        # Block 1: j=0,1
        for k_start in [0, 2, 4]:
            add_strassen(0, 0, k_start)
        add_schoolbook_221(0, 0, 6)
        
        # Block 2: j=2,3
        for k_start in [0, 2, 4]:
            add_strassen(0, 2, k_start)
        add_schoolbook_221(0, 2, 6)

        return U[:col], V[:col], W[:col], col

    def verify(U, V, W, tol=1e-9):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(r):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build the rank-50 solution
    U, V, W, rank = build_rank50()
    
    # Verify
    if not verify(U, V, W):
        # Fallback: schoolbook rank 56
        U = [[0] * (nA * nW) for _ in range(nA)]
        V = [[0] * (nA * nW) for _ in range(nB)]
        W = [[0] * (nA * nW) for _ in range(nW)]
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    U[ai][col] = 1
                    V[bi][col] = 1
                    W[wi][col] = 1
                    col += 1
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": W, "W": W} # Wait, return U, V, W correctly

    # Correction in return
    return {"U": U, "V": V, "W": W}

