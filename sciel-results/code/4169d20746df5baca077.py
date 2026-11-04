"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
  1. Build a rank-77 decomposition using top-level Strassen (7 terms) where each
     term is a (2x2) x (2x3) product. Each such product uses inner Strassen for
     the 2x2x2 part (7 terms) plus schoolbook for the 2x2x1 part (4 terms) = 11.
     Total rank: 7 * 11 = 77.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: A00 + A11
    {0: 0, 1: 0, 2: 1, 3: 1},   # M2: A10 + A11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00 + A01
    {0: -1, 1: 0, 2: 1, 3: 0},  # M6: A10 - A00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M7: A01 - A11
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: B00 + B11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M3: B01 - B11
    {0: -1, 1: 0, 2: 1, 3: 0},  # M4: B10 - B00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M6: B00 + B01
    {0: 0, 1: 0, 2: 1, 3: 1},   # M7: B10 + B11
]
recon = [
    [(0, 0, 1), (1, 1, 1)],       # M1 -> C00(+1), C11(+1)
    [(1, 0, 1), (1, 1, -1)],      # M2 -> C10(+1), C11(-1)
    [(0, 1, 1), (1, 1, 1)],       # M3 -> C01(+1), C11(+1)
    [(0, 0, 1), (1, 0, 1)],       # M4 -> C00(+1), C10(+1)
    [(0, 0, -1), (0, 1, 1)],      # M5 -> C00(-1), C01(+1)
    [(1, 1, 1)],                  # M6 -> C11(+1)
    [(0, 0, 1)],                  # M7 -> C00(+1)
]


def _build_rank77():
    """Build the rank-77 decomposition using top-level Strassen with 2x2x2+2x2x1 inner."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    R = 77
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si].get(a, 0) != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s].get(r, 0) != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * m + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if beta[si].get(b, 0) != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s].get(c, 0) != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * p + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * n + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s].get(r, 0) != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * m + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s].get(c, 0) != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * p + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * n + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    assert col == R, f"Column mismatch: {col} != {R}"
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build rank-77 decomposition
    U77, V77, W77 = _build_rank77()
    
    return {"U": U77.tolist(), "V": V77.tolist(), "W": W77.tolist()}
