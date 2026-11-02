"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Use Strassen's algorithm for 2x2x2 (rank 7) applied to 12 sub-products
    # A is 4x4, B is 4x6, C = A*B is 4x6
    # Split into 2x2 blocks:
    # A has 2x2 = 4 blocks of size 2x2
    # B has 2x3 = 6 blocks of size 2x2
    # C has 2x3 = 6 blocks of size 2x2
    # Each C block = sum of 2 products of A and B blocks
    # Total: 12 2x2*2x2 products, each rank 7 = 84 total rank
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen's 7 products for 2x2*2x2
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    strassen_x = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [0, 0, 1, -1],  # M6
        [0, 1, 0, -1],  # M7
    ]
    strassen_y = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    z_coeffs = [
        [1, 0, 0, 0, -1, 0, 1],   # Z00
        [0, 0, 1, 0, 1, 0, 0],    # Z01
        [0, 1, 0, 1, 0, 0, 0],    # Z10
        [1, -1, 1, 0, 0, 1, 0],   # Z11
    ]
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for bi in range(2):  # A row block (0 or 1)
        for bj in range(2):  # A col block / B row block (0 or 1)
            for bk in range(3):  # B col block (0, 1, or 2)
                # X = A[2*bi:2*bi+2, 2*bj:2*bj+2]
                # Y = B[2*bj:2*bj+2, 2*bk:2*bk+2]
                # Z = C[2*bi:2*bi+2, 2*bk:2*bk+2]
                
                x_indices = [
                    2*bi * 4 + 2*bj,      # X[0][0]
                    2*bi * 4 + 2*bj + 1,  # X[0][1]
                    (2*bi+1) * 4 + 2*bj,  # X[1][0]
                    (2*bi+1) * 4 + 2*bj + 1,  # X[1][1]
                ]
                y_indices = [
                    2*bj * 6 + 2*bk,      # Y[0][0]
                    2*bj * 6 + 2*bk + 1,  # Y[0][1]
                    (2*bj+1) * 6 + 2*bk,  # Y[1][0]
                    (2*bj+1) * 6 + 2*bk + 1,  # Y[1][1]
                ]
                z_indices = [
                    2*bk * 4 + 2*bi,      # Z[0][0]
                    (2*bk+1) * 4 + 2*bi,  # Z[0][1]
                    2*bk * 4 + 2*bi + 1,  # Z[1][0]
                    (2*bk+1) * 4 + 2*bi + 1,  # Z[1][1]
                ]
                
                for s in range(7):
                    for xi in range(4):
                        if strassen_x[s][xi] != 0:
                            U[x_indices[xi]][col] = strassen_x[s][xi]
                    for yi in range(4):
                        if strassen_y[s][yi] != 0:
                            V[y_indices[yi]][col] = strassen_y[s][yi]
                    for zi in range(4):
                        if z_coeffs[zi][s] != 0:
                            W[z_indices[zi]][col] = z_coeffs[zi][s]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
