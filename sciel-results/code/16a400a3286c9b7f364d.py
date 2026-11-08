"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen, rank 7."""
    factors = []
    # M1 = (A00+A11)(B00+B11)
    factors.append({
        'U': [(0, 1), (3, 1)],
        'V': [(0, 1), (3, 1)],
        'W': [(0, 1), (2, 1)],
    })
    # M2 = (A10+A11)B00
    factors.append({
        'U': [(2, 1), (3, 1)],
        'V': [(0, 1)],
        'W': [(2, -1), (3, -1)],
    })
    # M3 = A00(B01-B11)
    factors.append({
        'U': [(0, 1)],
        'V': [(1, 1), (3, -1)],
        'W': [(1, 1), (3, 1)],
    })
    # M4 = A11(B10-B00)
    factors.append({
        'U': [(3, 1)],
        'V': [(2, 1), (0, -1)],
        'W': [(0, -1), (1, 1)],
    })
    # M5 = (A00+A01)B11
    factors.append({
        'U': [(0, 1), (1, 1)],
        'V': [(3, 1)],
        'W': [(0, 1), (3, 1)],
    })
    # M6 = (A10-A00)(B00+B01)
    factors.append({
        'U': [(2, 1), (0, -1)],
        'V': [(0, 1), (1, 1)],
        'W': [(3, 1)],
    })
    # M7 = (A01-A11)(B10+B11)
    factors.append({
        'U': [(1, 1), (3, -1)],
        'V': [(2, 1), (3, 1)],
        'W': [(0, 1)],
    })
    
    U = [[0] * 7 for _ in range(8)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    for t, f in enumerate(factors):
        for row, val in f['U']:
            U[row][t] = val
        for row, val in f['V']:
            V[row][t] = val
        for row, val in f['W']:
            W[row][t] = val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s, V_s, W_s = _strassen_2x2()
    
    for a in range(2):
        for b in range(4):
            base_col = (a * 4 + b) * 7
            
            for i in range(2):
                for j in range(2):
                    full_row = i * m + (2 * a + j)
                    sub_row = i * 2 + j
                    for c in range(7):
                        if U_s[sub_row][c] != 0:
                            U[full_row][base_col + c] = U_s[sub_row][c]
            
            for j in range(2):
                for k in range(2):
                    full_row = (2 * a + j) * p + (2 * b + k)
                    sub_row = j * 2 + k
                    for c in range(7):
                        if V_s[sub_row][c] != 0:
                            V[full_row][base_col + c] = V_s[sub_row][c]
            
            for k in range(2):
                for i in range(2):
                    full_row = (2 * b + k) * n + i
                    sub_row = k * 2 + i
                    for c in range(7):
                        if W_s[sub_row][c] != 0:
                            W[full_row][base_col + c] = W_s[sub_row][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
