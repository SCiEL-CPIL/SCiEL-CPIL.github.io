import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # M1: (x11+x22)(y11+y22)
    # M2: (x21+x22)y11
    # M3: x11(y12-y22)
    # M4: x22(y21-y11)
    # M5: (x11+x12)y22
    # M6: (x21-x11)(y11+y12)
    # M7: (x12-x22)(y21+y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Block definitions: (A_indices, B_indices, C_indices)
    # A_indices: (x11, x12, x21, x22)
    # B_indices: (y11, y12, y21, y22)
    # C_indices: (z11, z12, z21, z22)
    
    blocks = []
    
    # A_L indices
    A_L = (0, 1, 4, 5)
    # A_R indices
    A_R = (2, 3, 6, 7)
    
    # B_T1 indices
    B_T1 = (0, 1, 6, 7)
    # B_T2 indices
    B_T2 = (2, 3, 8, 9)
    # B_T3 indices
    B_T3 = (4, 5, 10, 11)
    
    # B_B1 indices
    B_B1 = (12, 13, 18, 19)
    # B_B2 indices
    B_B2 = (14, 15, 20, 21)
    # B_B3 indices
    B_B3 = (16, 17, 22, 23)
    
    # C indices for k=0,1
    C_01 = (0, 2, 1, 3)
    # C indices for k=2,3
    C_23 = (4, 6, 5, 7)
    # C indices for k=4,5
    C_45 = (8, 10, 9, 11)
    
    blocks.append((A_L, B_T1, C_01))
    blocks.append((A_L, B_T2, C_23))
    blocks.append((A_L, B_T3, C_45))
    blocks.append((A_R, B_B1, C_01))
    blocks.append((A_R, B_B2, C_23))
    blocks.append((A_R, B_B3, C_45))
    
    col = 0
    for A_idx, B_idx, C_idx in blocks:
        x11, x12, x21, x22 = A_idx
        y11, y12, y21, y22 = B_idx
        z11, z12, z21, z22 = C_idx
        
        # M1
        U[x11][col] += 1
        U[x22][col] += 1
        V[y11][col] += 1
        V[y22][col] += 1
        W[z11][col] += 1
        W[z22][col] += 1
        col += 1
        
        # M2
        U[x21][col] += 1
        U[x22][col] += 1
        V[y11][col] += 1
        W[z21][col] += 1
        W[z22][col] -= 1
        col += 1
        
        # M3
        U[x11][col] += 1
        V[y12][col] += 1
        V[y22][col] -= 1
        W[z12][col] += 1
        W[z22][col] += 1
        col += 1
        
        # M4
        U[x22][col] += 1
        V[y21][col] += 1
        V[y11][col] -= 1
        W[z11][col] += 1
        W[z21][col] += 1
        col += 1
        
        # M5
        U[x11][col] += 1
        U[x12][col] += 1
        V[y22][col] += 1
        W[z11][col] -= 1
        W[z12][col] += 1
        col += 1
        
        # M6
        U[x21][col] += 1
        U[x11][col] -= 1
        V[y11][col] += 1
        V[y12][col] += 1
        W[z22][col] += 1
        col += 1
        
        # M7
        U[x12][col] += 1
        U[x22][col] -= 1
        V[y21][col] += 1
        V[y22][col] += 1
        W[z11][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
