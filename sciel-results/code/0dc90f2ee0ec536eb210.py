import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Strassen definitions
    # M1: (a00+a11)(b00+b11) -> C00+1, C11+1
    # M2: (a10+a11)b00 -> C10+1, C11-1
    # M3: a00(b01-b11) -> C01+1, C11+1
    # M4: a11(b10-b00) -> C00+1, C10+1
    # M5: (a00+a01)b11 -> C00-1, C01+1
    # M6: (a10-a00)(b00+b01) -> C11+1
    # M7: (a01-a11)(b10+b11) -> C00+1
    
    strassen_steps = [
        # (u_coeffs [(i,j,val)], v_coeffs [(j,k,val)], w_coeffs [(i,k,val)])
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]), # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]), # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]), # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]), # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]), # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]), # M7
    ]
    
    # Iterate over blocks
    # r_B: row block of B (0,1)
    # c_B: col block of B (0,1,2,3)
    for r_B in range(2):
        for c_B in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
                # Fill U
                for i, j, val in u_coeffs:
                    # Global A index: i * m + (2*r_B + j)
                    u_idx = i * m + (2 * r_B + j)
                    U[u_idx, col] += val
                # Fill V
                for j, k, val in v_coeffs:
                    # Global B index: (2*r_B + j) * p + (2*c_B + k)
                    v_idx = (2 * r_B + j) * p + (2 * c_B + k)
                    V[v_idx, col] += val
                # Fill W
                for i, k, val in w_coeffs:
                    # Global C index: (2*c_B + k) * n + i
                    w_idx = (2 * c_B + k) * n + i
                    W[w_idx, col] += val
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
