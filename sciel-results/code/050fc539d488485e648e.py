import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U coeffs for A (4 entries: 00, 01, 10, 11)
    # V coeffs for B (4 entries: 00, 01, 10, 11)
    # W coeffs for C (4 entries: 00, 01, 10, 11)
    
    # M1: (a00+a11)(b00+b11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: c00 += 1, c01 += 0, c10 += 0, c11 += 1 -> [1, 0, 0, 1]
    
    # M2: (a10+a11)(b00+b01)
    # U: [0, 0, 1, 1]
    # V: [1, 1, 0, 0]
    # W: c00 += -1, c01 += 1, c10 += 0, c11 += 0 -> [-1, 1, 0, 0]
    
    # M3: (a00-a10)(b10-b11)
    # U: [1, 0, -1, 0]
    # V: [0, 0, 1, -1]
    # W: c00 += 0, c01 += 0, c10 += 1, c11 += -1 -> [0, 0, 1, -1]
    
    # M4: a00(b01-b11)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: c00 += 1, c01 += -1, c10 += 0, c11 += 0 -> [1, -1, 0, 0]
    
    # M5: a11(b00-b10)
    # U: [0, 0, 0, 1]
    # V: [1, 0, -1, 0]
    # W: c00 += -1, c01 += 0, c10 += 1, c11 += 0 -> [-1, 0, 1, 0]
    
    # M6: (a00+a01)(b11-b00)
    # U: [1, 1, 0, 0]
    # V: [-1, 0, 0, 1]
    # W: c00 += 0, c01 += -1, c10 += 1, c11 += -1 -> [0, -1, 1, -1]
    
    # M7: (a10+a01)(b00+b11)
    # U: [0, 1, 1, 0]
    # V: [1, 0, 0, 1]
    # W: c00 += 1, c01 += -1, c10 += 1, c11 += 0 -> [1, -1, 1, 0]
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0]
    ]
    
    strassen_V = [
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1],
        [1, 0, -1, 0],
        [-1, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    
    strassen_W = [
        [1, 0, 0, 1],
        [-1, 1, 0, 0],
        [0, 0, 1, -1],
        [1, -1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 1, -1],
        [1, -1, 1, 0]
    ]
    
    # Fill U for both blocks (same U)
    for r in range(7):
        for i in range(4):
            U[i][r] = strassen_U[r][i]
            U[i][r+7] = strassen_U[r][i]
            
    # Fill V for first block (cols 0,1)
    # V index: j*p + k. p=4.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    for r in range(7):
        v_coeffs = strassen_V[r]
        # v_coeffs[0] is b00 (j=0, k=0)
        V[0*4 + 0][r] = v_coeffs[0]
        V[0*4 + 1][r] = v_coeffs[1]
        V[1*4 + 0][r] = v_coeffs[2]
        V[1*4 + 1][r] = v_coeffs[3]
        
    # Fill V for second block (cols 2,3)
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 6
    # j=1, k=3 -> 7
    for r in range(7):
        v_coeffs = strassen_V[r]
        V[0*4 + 2][r+7] = v_coeffs[0]
        V[0*4 + 3][r+7] = v_coeffs[1]
        V[1*4 + 2][r+7] = v_coeffs[2]
        V[1*4 + 3][r+7] = v_coeffs[3]
        
    # Fill W for first block (cols 0,1)
    # W index: k*n + i. n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    for r in range(7):
        w_coeffs = strassen_W[r]
        # w_coeffs[0] is c00 (k=0, i=0)
        W[0*2 + 0][r] = w_coeffs[0]
        W[0*2 + 1][r] = w_coeffs[1]
        W[1*2 + 0][r] = w_coeffs[2]
        W[1*2 + 1][r] = w_coeffs[3]
        
    # Fill W for second block (cols 2,3)
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    for r in range(7):
        w_coeffs = strassen_W[r]
        W[2*2 + 0][r+7] = w_coeffs[0]
        W[2*2 + 1][r+7] = w_coeffs[1]
        W[3*2 + 0][r+7] = w_coeffs[2]
        W[3*2 + 1][r+7] = w_coeffs[3]
        
    return {"U": U, "V": V, "W": W}
