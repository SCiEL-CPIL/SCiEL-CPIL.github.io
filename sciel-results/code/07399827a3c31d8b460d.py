import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # Gates 0..6
    # U coeffs (for x_00, x_01, x_10, x_11)
    strassen_u = [
        [1, 0, 0, 1], # M0: x00 + x11
        [0, 0, 1, 1], # M1: x10 + x11
        [1, 0, 0, 0], # M2: x00
        [0, 0, 0, 1], # M3: x11
        [1, 0, 1, 0], # M4: x00 + x10
        [-1, 0, 1, 0],# M5: x10 - x00
        [0, -1, 0, 1] # M6: x11 - x01
    ]
    
    # V coeffs (for y_00, y_01, y_10, y_11)
    strassen_v = [
        [1, 0, 0, 1], # M0: y00 + y11
        [1, 0, 0, 0], # M1: y00
        [0, -1, 1, 0],# M2: y10 - y11
        [-1, 1, 0, 0],# M3: y01 - y00
        [0, 0, 0, 1], # M4: y11
        [1, 0, 1, 0], # M5: y00 + y10
        [0, 1, 0, 1]  # M6: y01 + y11
    ]
    
    # W coeffs (for z_00, z_01, z_10, z_11)
    # z00 = M0 + M3 - M4 + M6
    # z01 = M2 + M4
    # z10 = M1 + M3
    # z11 = M0 - M1 + M2 + M5
    strassen_w = [
        [1, 0, 0, -1], # M0 contributes to z00, z11
        [0, 0, 1, -1], # M1 contributes to z10, z11
        [0, 1, 0, 1],  # M2 contributes to z01, z11
        [1, 0, 1, 0],  # M3 contributes to z00, z10
        [-1, 1, 0, 0], # M4 contributes to z00, z01
        [0, 0, 0, 1],  # M5 contributes to z11
        [1, 0, 0, 0]   # M6 contributes to z00
    ]
    
    # Mapping of x,y indices to U,V indices for a given block b and sub-instance s
    # x indices: 0,1,2,3 correspond to x00, x01, x10, x11
    # y indices: 0,1,2,3 correspond to y00, y01, y10, y11
    
    for g in range(rank):
        b = g // 14
        s = (g // 7) % 2
        m_idx = g % 7
        
        # U indices
        if s == 0:
            # A1: rows 0,1
            u_idx_map = [0, 1, 4, 5]
        else:
            # A2: rows 2,3
            u_idx_map = [8, 9, 12, 13]
            
        # V indices
        # B_b: cols 2b, 2b+1
        if s == 0:
            # Top: rows 0,1
            v_idx_map = [2*b, 2*b+1, 8+2*b, 8+2*b+1]
        else:
            # Bottom: rows 2,3
            v_idx_map = [16+2*b, 16+2*b+1, 24+2*b, 24+2*b+1]
            
        # W indices
        # C_b: cols 2b, 2b+1, rows 0,1
        # W[k*n + i]
        w_idx_map = [4*b, 4*b+1, 4*b+2, 4*b+3]
        
        # Fill U
        for k in range(4):
            val = strassen_u[m_idx][k]
            if val != 0:
                U[u_idx_map[k]][g] = val
                
        # Fill V
        for k in range(4):
            val = strassen_v[m_idx][k]
            if val != 0:
                V[v_idx_map[k]][g] = val
                
        # Fill W
        for k in range(4):
            val = strassen_w[m_idx][k]
            if val != 0:
                W[w_idx_map[k]][g] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
