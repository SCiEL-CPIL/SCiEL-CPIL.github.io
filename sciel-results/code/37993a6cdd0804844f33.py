import time
import random
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 4
    rank = 49  # 7 * 7: outer Strassen x inner Strassen
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard 7-term Strassen forms for 2x2:
    # Each form is a dict mapping (row, col) -> coefficient
    
    # A-forms (linear combinations of A entries)
    A_forms = [
        {(0,0): 1, (1,1): 1},    # M0: A00 + A11
        {(1,0): 1, (1,1): 1},    # M1: A10 + A11
        {(0,0): 1},              # M2: A00
        {(1,1): 1},              # M3: A11
        {(0,0): 1, (0,1): 1},    # M4: A00 + A01
        {(1,0): 1, (0,0): -1},   # M5: A10 - A00
        {(0,1): 1, (1,1): -1},   # M6: A01 - A11
    ]
    
    # B-forms (linear combinations of B entries)
    B_forms = [
        {(0,0): 1, (1,1): 1},    # M0: B00 + B11
        {(0,0): 1},              # M1: B00
        {(0,1): 1, (1,1): -1},   # M2: B01 - B11
        {(1,0): 1, (0,0): -1},   # M3: B10 - B00
        {(1,1): 1},              # M4: B11
        {(0,0): 1, (0,1): 1},    # M5: B00 + B01
        {(1,0): 1, (1,1): 1},    # M6: B10 + B11
    ]
    
    # C-forms (linear combinations of C entries, indexed by output position)
    # C[i,k] gets contribution from M_s
    C_forms = [
        {(0,0): 1, (1,1): 1},    # M0: C00 + C11
        {(1,0): 1, (1,1): -1},   # M1: +C10, -C11
        {(0,1): 1},              # M2: +C01
        {(1,0): 1, (0,0): -1},   # M3: +C10, -C00
        {(0,0): 1, (0,1): 1},    # M4: +C00, +C01
        {(1,1): 1},              # M5: +C11
        {(0,0): 1},              # M6: +C00
    ]
    
    # Build 49-rank decomposition via tensor product of two Strassen
    # 4x4 matrix = 2x2 blocks of 2x2
    # For pair (s, t) in 0..6 x 0..6:
    #   col = 7*s + t
    #   U[i*4+j, col] = sum over macro (S,T) and micro (i2,j2):
    #       A_forms[s][(S,T)] * A_forms[t][(i2,j2)]
    #       where i = 2*S + i2, j = 2*T + j2
    
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            
            # Fill U
            for (Si, Ti), coeff_outer in A_forms[s].items():
                if coeff_outer == 0:
                    continue
                for (i2, j2), coeff_inner in A_forms[t].items():
                    if coeff_inner == 0:
                        continue
                    i = 2 * Si + i2
                    j = 2 * Ti + j2
                    U[i * 4 + j][col] += coeff_outer * coeff_inner
            
            # Fill V: V[j*4+k, col]
            # j = 2*T + j2, k = 2*K + k2
            for (Ti, Ki), coeff_outer in B_forms[s].items():
                if coeff_outer == 0:
                    continue
                for (j2, k2), coeff_inner in B_forms[t].items():
                    if coeff_inner == 0:
                        continue
                    j = 2 * Ti + j2
                    k = 2 * Ki + k2
                    V[j * 4 + k][col] += coeff_outer * coeff_inner
            
            # Fill W: W[k*4+i, col]  (trace convention)
            # i = 2*S + i2, k = 2*K + k2
            for (Si, Ki), coeff_outer in C_forms[s].items():
                if coeff_outer == 0:
                    continue
                for (i2, k2), coeff_inner in C_forms[t].items():
                    if coeff_inner == 0:
                        continue
                    i = 2 * Si + i2
                    k = 2 * Ki + k2
                    W[k * 4 + i][col] += coeff_outer * coeff_inner
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
