"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen algorithm for each 2x2 block of B
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Predefine the u-vectors for the 7 Strassen steps
    # Indices: 0=a00, 1=a01, 2=a10, 3=a11
    u_defs = [
        [1, 0, 0, 1],   # M0: a00 + a11
        [0, 0, 1, 1],   # M1: a10 + a11
        [1, 0, 0, 0],   # M2: a00
        [0, 0, 0, 1],   # M3: a11
        [1, 1, 0, 0],   # M4: a00 + a01
        [-1, 0, 1, 0],  # M5: a10 - a00
        [0, 1, 0, -1]   # M6: a01 - a11
    ]
    
    # For each of the 4 blocks (l=0..3), and each of the 7 steps (r=0..6)
    for l in range(4):
        for r in range(7):
            col = l * 7 + r
            
            # Fill U
            for i in range(4):
                U[i][col] = u_defs[r][i]
                
            # Fill V
            # V indices: 8*j + k
            # Block l uses columns 2l and 2l+1
            # b00 is at (0, 2l) -> idx 2l
            # b01 is at (0, 2l+1) -> idx 2l+1
            # b10 is at (1, 2l) -> idx 8+2l
            # b11 is at (1, 2l+1) -> idx 9+2l
            
            # Fill W
            # W indices: 2*k + i
            # Block l outputs to columns 2l and 2l+1
            # C00 is at (0, 2l) -> idx 4l
            # C01 is at (0, 2l+1) -> idx 4l+2
            # C10 is at (1, 2l) -> idx 4l+1
            # C11 is at (1, 2l+1) -> idx 4l+3
            
            if r == 0:
                # M0 = (a00+a11)(b00+b11)
                # V: b00 + b11
                V[2*l][col] = 1
                V[9+2*l][col] = 1
                # W: C00 += M0, C11 += M0
                W[4*l][col] = 1
                W[4*l+3][col] = 1
            elif r == 1:
                # M1 = (a10+a11)b00
                # V: b00
                V[2*l][col] = 1
                # W: C10 += M1, C11 -= M1
                W[4*l+1][col] = 1
                W[4*l+3][col] = -1
            elif r == 2:
                # M2 = a00(b01-b11)
                # V: b01 - b11
                V[2*l+1][col] = 1
                V[9+2*l][col] = -1
                # W: C01 += M2
                W[4*l+2][col] = 1
            elif r == 3:
                # M3 = a11(b00-b01)
                # V: b00 - b01
                V[2*l][col] = 1
                V[2*l+1][col] = -1
                # W: C00 -= M3, C11 += M3
                W[4*l][col] = -1
                W[4*l+3][col] = 1
            elif r == 4:
                # M4 = (a00+a01)b11
                # V: b11
                V[9+2*l][col] = 1
                # W: C00 += M4
                W[4*l][col] = 1
            elif r == 5:
                # M5 = (a10-a00)(b00+b01)
                # V: b00 + b01
                V[2*l][col] = 1
                V[2*l+1][col] = 1
                # W: C10 += M5, C11 += M5
                W[4*l+1][col] = 1
                W[4*l+3][col] = 1
            elif r == 6:
                # M6 = (a01-a11)(b10+b11)
                # V: b10 + b11
                V[8+2*l][col] = 1
                V[9+2*l][col] = 1
                # W: C00 -= M6, C01 += M6
                W[4*l][col] = -1
                W[4*l+2][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
