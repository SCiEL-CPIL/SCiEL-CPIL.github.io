"""Matrix multiplication 3x4 by 4x6 - recursive block decomposition."""

import numpy as np

def _strassen_decomp():
    """Returns U, V, W for 2x2x2 Strassen decomposition."""
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    # Note: U is (n*m, rank) = (4, 7)
    # My list above is (7, 4). I need to transpose.
    
    V = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, -1, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    W = [
        [1, 0, 0, 0],  # C00
        [0, 1, 0, 0],  # C01
        [0, 0, 1, 0],  # C10
        [0, 0, 0, 1]   # C11
    ]
    # W is (p*n, rank) = (4, 7).
    # The coefficients for C:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W[c_idx, m_idx]
    W_coeff = [
        [1, 0, 0, -1, 1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Convert to lists of lists with correct orientation
    # U: (4, 7)
    U_out = [[0]*7 for _ in range(4)]
    for r in range(7):
        for c in range(4):
            U_out[c][r] = U[r][c]
            
    # V: (4, 7)
    V_out = [[0]*7 for _ in range(4)]
    for r in range(7):
        for c in range(4):
            V_out[c][r] = V[r][c]
            
    # W: (4, 7)
    W_out = [[0]*7 for _ in range(4)]
    for r in range(4):
        for c in range(7):
            W_out[r][c] = W_coeff[r][c]
            
    return U_out, V_out, W_out

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_decomp(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen_decomp()
    
    if m == 1 or n == 1 or p == 1:
        return _schoolbook_decomp(n, m, p)
        
    if m % 2 == 0:
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1 = _build_decomp(n, m1, p)
        U2, V2, W2 = _build_decomp(n, m2, p)
        
        rank = len(U1[0]) + len(U2[0])
        U = [row[:] for row in U1] + [row[:] for row in U2]
        V = [row[:] for row in V1] + [row[:] for row in V2]
        
        W = [[0] * rank for _ in range(p * n)]
        for i in range(p * n):
            for j in range(len(W1[0])):
                W[i][j] = W1[i][j]
            for j in range(len(W2[0])):
                W[i][len(W1[0]) + j] = W2[i][j]
                
        return U, V, W
        
    if p % 2 == 0:
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1 = _build_decomp(n, m, p1)
        U2, V2, W2 = _build_decomp(n, m, p2)
        
        rank = len(U1[0]) + len(U2[0])
        U = [[0] * rank for _ in range(n * m)]
        for i in range(n * m):
            for j in range(len(U1[0])):
                U[i][j] = U1[i][j]
            for j in range(len(U2[0])):
                U[i][len(U1[0]) + j] = U2[i][j]
                
        V = [row[:] for row in V1] + [row[:] for row in V2]
        W = [row[:] for row in W1] + [row[:] for row in W2]
        
        return U, V, W
        
    if n % 2 == 0:
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1 = _build_decomp(n1, m, p)
        U2, V2, W2 = _build_decomp(n2, m, p)
        
        rank = len(U1[0]) + len(U2[0])
        U = [row[:] for row in U1] + [row[:] for row in U2]
        
        V = [[0] * rank for _ in range(m * p)]
        for i in range(m * p):
            for j in range(len(V1[0])):
                V[i][j] = V1[i][j]
            for j in range(len(V2[0])):
                V[i][len(V1[0]) + j] = V2[i][j]
                
        W = [row[:] for row in W1] + [row[:] for row in W2]
        
        return U, V, W
        
    # Fallback
    return _schoolbook_decomp(n, m, p)

def construct():
    n, m, p = 3, 4, 6
    U, V, W = _build_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
