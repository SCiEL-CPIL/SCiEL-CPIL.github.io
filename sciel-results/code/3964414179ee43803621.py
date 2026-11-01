"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Strassen's 7 bilinear forms for 2x2 x 2x2
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,1):1}),
        # M2 = (a10+a11)b00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = a00(b01-b11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = a11(b10-b00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (a00+a01)b11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (a10-a00)(b00+b01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (a01-a11)(b10+b11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Schoolbook terms for 2x2 * 2x1 (single column)
    schoolbook_2x2x1 = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(1,0):1}, {(0,0):1}, {(1,0):1}),
        ({(1,1):1}, {(1,0):1}, {(1,0):1}),
    ]
    
    rank = 7 * 7 + 4  # 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process 7 blocks of 2 columns each
    for block in range(7):
        k_start = block * 2
        for s in range(7):
            A_coeffs, B_coeffs, C_coeffs = strassen_terms[s]
            
            for (i, j), val in A_coeffs.items():
                U[i * m + j][col] = val
            
            for (j, k_local), val in B_coeffs.items():
                k_global = k_start + k_local
                V[j * p + k_global][col] = val
            
            for (i_out, k_local), val in C_coeffs.items():
                k_global = k_start + k_local
                W[k_global * n + i_out][col] = val
            
            col += 1
    
    # Process last single column (column 14)
    k_start = 14
    for s in range(4):
        A_coeffs, B_coeffs, C_coeffs = schoolbook_2x2x1[s]
        
        for (i, j), val in A_coeffs.items():
            U[i * m + j][col] = val
        
        for (j, k_local), val in B_coeffs.items():
            k_global = k_start + k_local
            V[j * p + k_global][col] = val
        
        for (i_out, k_local), val in C_coeffs.items():
            k_global = k_start + k_local
            W[k_global * n + i_out][col] = val
        
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
