import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks
    # A1 is cols 0,1 of A. Rows 0,1.
    # U indices for A1:
    # x00: U[0, col]
    # x01: U[1, col]
    # x10: U[3, col]
    # x11: U[4, col]
    
    # Define Strassen terms for a generic 2x2 block
    # Returns list of (u_indices, v_indices, w_coefficients)
    # u_indices: list of (row, coeff)
    # v_indices: list of (row, coeff)
    # w_coefficients: dict mapping w_row to coeff
    
    def strassen_block(v_base_col, w_base_row):
        # v_base_col: offset for V rows (0, 6, or 12)
        # w_base_row: offset for W rows (0, 4, or 8)
        
        # M1 = (x00 + x11)(y00 + y11)
        u1 = [(0, 1), (4, 1)]
        v1 = [(v_base_col + 0, 1), (v_base_col + 7, 1)] # y00 is V[0, col], y11 is V[7, col] relative to block start?
        # Wait, V indices depend on j and k.
        # Block 1: j in 0,1. k in 0,1.
        # y00: j=0, k=0 -> V[0*6+0] = V[0]
        # y11: j=1, k=1 -> V[1*6+1] = V[7]
        # Block 2: j in 0,1. k in 2,3.
        # y00: j=0, k=2 -> V[2]
        # y11: j=1, k=3 -> V[9]
        # Block 3: j in 0,1. k in 4,5.
        # y00: j=0, k=4 -> V[4]
        # y11: j=1, k=5 -> V[11]
        
        # So v indices are not just offset.
        # Let's pass the actual V row indices for y00, y01, y10, y11.
        
        pass

    # Let's just hardcode the indices for clarity and correctness.
    
    # Block 1: k in 0,1
    # V rows: y00=0, y01=1, y10=6, y11=7
    # W rows: z00=0, z01=2, z10=1, z11=3
    
    # Block 2: k in 2,3
    # V rows: y00=2, y01=3, y10=8, y11=9
    # W rows: z00=4, z01=6, z10=5, z11=7
    
    # Block 3: k in 4,5
    # V rows: y00=4, y01=5, y10=10, y11=11
    # W rows: z00=8, z01=10, z10=9, z11=11

    blocks = [
        {
            'v': [0, 1, 6, 7], # y00, y01, y10, y11
            'w': [0, 2, 1, 3]  # z00, z01, z10, z11
        },
        {
            'v': [2, 3, 8, 9],
            'w': [4, 6, 5, 7]
        },
        {
            'v': [4, 5, 10, 11],
            'w': [8, 10, 9, 11]
        }
    ]

    # U rows for A1: x00=0, x01=1, x10=3, x11=4
    u_x00 = 0
    u_x01 = 1
    u_x10 = 3
    u_x11 = 4

    for block in blocks:
        v00, v01, v10, v11 = block['v']
        w00, w01, w10, w11 = block['w']
        
        # M1
        U[u_x00][col] += 1
        U[u_x11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (x10 + x11) y00
        U[u_x10][col] += 1
        U[u_x11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M3 = x00 (y01 - y11)
        U[u_x00][col] += 1
        V[v01][col] += 1
        V[v11][col] += -1
        W[w01][col] += 1
        col += 1
        
        # M4 = x11 (y10 - y00)
        U[u_x11][col] += 1
        V[v10][col] += 1
        V[v00][col] += -1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M5 = (x00 + x01) y11
        U[u_x00][col] += 1
        U[u_x01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1
        W[w01][col] += 1
        col += 1
        
        # M6 = (x10 - x00)(y00 + y01)
        U[u_x10][col] += 1
        U[u_x00][col] += -1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (x01 - x11)(y10 + y11)
        U[u_x01][col] += 1
        U[u_x11][col] += -1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += -1
        col += 1

    # Scalar terms
    # A2: x0=2, x1=5
    # B2: k in 0..5 -> V rows 12..17
    # C: k in 0..5
    # C0k: W row 2k
    # C1k: W row 2k+1
    
    for k in range(6):
        # A02 * B2k
        U[2][col] += 1
        V[12+k][col] += 1
        W[2*k][col] += 1
        col += 1
        
        # A12 * B2k
        U[5][col] += 1
        V[12+k][col] += 1
        W[2*k+1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
