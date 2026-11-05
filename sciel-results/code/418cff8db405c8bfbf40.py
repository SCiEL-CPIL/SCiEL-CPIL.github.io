import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0

    def rank36_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        # Correct Strassen terms for 2x2x2
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Indices for 2x2 A: 00=0, 01=1, 10=2, 11=3
        # Indices for 2x2 B: 00=0, 01=1, 10=2, 11=3
        # Indices for 2x2 C: 00=0, 01=1, 10=2, 11=3

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
            ({0: 1}, {1: 1, 3: -1}, {1: 1}),                  # M3
            ({3: 1}, {2: 1, 0: -1}, {0: -1, 2: 1}),           # M4
            ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1}),             # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1}),           # M7
        ]

        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        # Decompose M_2,4,5 into two M_2,2,5
        # A = [A1, A2] (2x2 each)
        # B = [B1; B2] (2x5 each)
        # C = A1 B1 + A2 B2
        
        for q in range(2):
            # B_q is 2x5. Split into B_q0 (2x2), B_q1 (2x2), B_q2 (2x1)
            for s in range(3):
                if s < 2:
                    # 2x2 * 2x2
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            # C is 2x5.
                            # Local C is 2x2.
                            # Local row lk, local col li.
                            # Global row is li (0 or 1).
                            # Global col is 2*s + lk.
                            # W index = row * n + col.
                            w_g[li * n + (2 * s + lk)] = val
                        add_term(u_g, v_g, w_g)
                else:
                    # 2x2 * 2x1
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[li * n + 4] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    sol = rank36_solution()
    if verify(sol["U"], sol["V"], sol["W"]):
        return sol
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    r = (i0 * m + j0) * p + k0
                    U[i0 * m + j0][r] = 1
                    V[j0 * p + k0][r] = 1
                    W[k0 * n + i0][r] = 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
