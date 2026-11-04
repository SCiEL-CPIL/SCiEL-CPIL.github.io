"""Improved program: top-level Strassen with inner Strassen+schoolbook for 4x4*4x6."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Top-level Strassen (7 terms) where each term is a (2x2) x (2x3) product.
    For each (2x2)x(2x3): use inner Strassen for the 2x2x2 part (7 mults) + 
    schoolbook for the 2x2x1 part (4 mults) = 11 mults per term.
    Total rank: 7 * 11 = 77.
    """
    n, m, p = 4, 4, 6
    
    # Top-level Strassen coefficients
    # A blocks indexed 0-3: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B blocks indexed 0-3: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},   # M1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0},  # M5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}   # M6: A01 - A11
    ]
    
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M1: B00
        {0: 0, 1: 1, 2: 0, 3: -1},  # M2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0},  # M3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}    # M6: B10 + B11
    ]
    
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # M0 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],    # M1 -> C10, -C11
        [(0, 1, 1), (1, 1, 1)],     # M2 -> C01, C11
        [(0, 0, 1), (1, 0, 1)],     # M3 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],    # M4 -> -C00, C01
        [(1, 1, 1)],                # M5 -> C11
        [(0, 0, 1)]                 # M6 -> C00
    ]
    
    # Inner Strassen for 2x2x2 (same structure)
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                coeff_a = inner_alpha[si].get(a, 0)
                if coeff_a != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        coeff_r = alpha[s].get(r, 0)
                        if coeff_r != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += coeff_a * coeff_r
            
            for b in range(4):
                coeff_b = inner_beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b_idx * 3 + zb) * 4 + (a_idx * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                
                for a_idx, b_idx, gamma in recon[s]:
                    w_idx = (b_idx * 3 + 2) * 4 + (a_idx * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Convert to lists of integers/floats
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
