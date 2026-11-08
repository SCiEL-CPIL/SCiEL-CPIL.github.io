import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen step
    # block_type: 1: A1 B11, 2: A1 B12, 3: A2 B21, 4: A2 B22
    # step: 1-7
    # We map to global indices
    
    def add_strassen(block_type, step):
        col = (block_type - 1) * 7 + (step - 1)
        
        # Define indices for X (A part) and Y (B part)
        # A1: U[0], U[1], U[4], U[5]
        # A2: U[2], U[3], U[6], U[7]
        # B11: V[0], V[1], V[4], V[5]
        # B12: V[2], V[3], V[6], V[7]
        # B21: V[8], V[9], V[12], V[13]
        # B22: V[10], V[11], V[14], V[15]
        
        if block_type == 1 or block_type == 2:
            x00, x01, x10, x11 = 0, 1, 4, 5
        else:
            x00, x01, x10, x11 = 2, 3, 6, 7
            
        if block_type == 1 or block_type == 3:
            y00, y01, y10, y11 = 0, 1, 4, 5
            # W indices for k=0,1 -> 0,1,2,3
            w00, w10, w01, w11 = 0, 1, 2, 3
        else:
            y00, y01, y10, y11 = 2, 3, 6, 7
            # W indices for k=2,3 -> 4,5,6,7
            w00, w10, w01, w11 = 4, 5, 6, 7
            
        # Strassen formulas
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X01+X11)Y00
        # P3 = X00(Y10-Y11)
        # P4 = X11(Y00-Y10)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X10+X11)(Y10-Y00)
        
        if step == 1:
            U[x00][col] += 1
            U[x11][col] += 1
            V[y00][col] += 1
            V[y11][col] += 1
            # Z00 += P1, Z11 += P1
            W[w00][col] += 1
            W[w11][col] += 1
        elif step == 2:
            U[x01][col] += 1
            U[x11][col] += 1
            V[y00][col] += 1
            # Z01 += P2
            W[w01][col] += 1
        elif step == 3:
            U[x00][col] += 1
            V[y10][col] += 1
            V[y11][col] -= 1
            # Z01 += P3, Z11 += P3
            W[w01][col] += 1
            W[w11][col] += 1
        elif step == 4:
            U[x11][col] += 1
            V[y00][col] += 1
            V[y10][col] -= 1
            # Z00 += P4, Z11 += P4
            W[w00][col] += 1
            W[w11][col] += 1
        elif step == 5:
            U[x00][col] += 1
            U[x01][col] += 1
            V[y11][col] += 1
            # Z00 -= P5
            W[w00][col] -= 1
        elif step == 6:
            U[x10][col] += 1
            U[x00][col] -= 1
            V[y00][col] += 1
            V[y01][col] += 1
            # Z10 += P6
            W[w10][col] += 1
        elif step == 7:
            U[x10][col] += 1
            U[x11][col] += 1
            V[y10][col] += 1
            V[y00][col] -= 1
            # Z00 += P7, Z10 += P7
            W[w00][col] += 1
            W[w10][col] += 1

    for b in range(1, 5):
        for s in range(1, 8):
            add_strassen(b, s)

    return {"U": U, "V": V, "W": W}
