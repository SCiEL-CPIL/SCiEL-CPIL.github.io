import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen on A1 (2x2) and B1 blocks (2x2)
    # A1 uses i in {0,1}, j in {0,1}
    # B1 uses j in {0,1}, k in {0..5}
    
    # Strassen coefficients for A
    # l=1: A00 + A11
    # l=2: A10 + A11
    # l=3: A00
    # l=4: A11
    # l=5: A00 + A01
    # l=6: A10 - A00
    # l=7: A01 - A11
    
    A_coeffs = [
        {0: 1, 4: 1},       # l=1
        {3: 1, 4: 1},       # l=2
        {0: 1},             # l=3
        {4: 1},             # l=4
        {0: 1, 1: 1},       # l=5
        {3: 1, 0: -1},      # l=6
        {1: 1, 4: -1}       # l=7
    ]
    
    # For each block q in 0,1,2
    for q in range(3):
        k_start = 2 * q
        # B indices for this block
        # B00 -> 2q, B01 -> 2q+1, B10 -> 6+2q, B11 -> 6+2q+1
        
        B_coeffs = [
            {2*q: 1, 6+2*q+1: 1},       # l=1
            {2*q: 1},                   # l=2
            {2*q+1: 1, 6+2*q+1: -1},    # l=3
            {6+2*q: 1, 2*q: -1},        # l=4
            {6+2*q+1: 1},               # l=5
            {2*q: 1, 2*q+1: 1},         # l=6
            {6+2*q: 1, 6+2*q+1: 1}      # l=7
        ]
        
        W_coeffs = [
            {2*q*2+0: 1, (2*q+1)*2+1: 1},      # l=1
            {2*q*2+1: 1, (2*q+1)*2+1: -1},     # l=2
            {(2*q+1)*2+0: 1, (2*q+1)*2+1: 1},  # l=3
            {2*q*2+0: -1, 2*q*2+1: 1},         # l=4
            {2*q*2+0: 1},                      # l=5
            {(2*q+1)*2+1: 1},                  # l=6
            {2*q*2+0: -1, (2*q+1)*2+0: 1}      # l=7
        ]
        
        for l in range(7):
            c = q * 7 + l
            for idx, val in A_coeffs[l].items():
                U[idx][c] += val
            for idx, val in B_coeffs[l].items():
                V[idx][c] += val
            for idx, val in W_coeffs[l].items():
                W[idx][c] += val

    # Part 2: A2 (2x1) and B2 (1x6)
    # A2 uses i in {0,1}, j=2
    # B2 uses j=2, k in {0..5}
    
    for k in range(6):
        # i=0
        c0 = 21 + 2 * k
        U[2][c0] = 1
        V[12 + k][c0] = 1
        W[k * 2 + 0][c0] = 1
        
        # i=1
        c1 = 22 + 2 * k
        U[5][c1] = 1
        V[12 + k][c1] = 1
        W[k * 2 + 1][c1] = 1

    return {"U": U, "V": V, "W": W}
