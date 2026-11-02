"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # Rank is 27 (3x3 block mults) * 7 (Strassen) = 189
    rank = 189
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    X_coeffs = [
        [[1, 0], [0, 1]], # P1: X00 + X11
        [[0, 0], [1, 1]], # P2: X10 + X11
        [[1, 0], [0, 0]], # P3: X00
        [[0, 0], [0, 1]], # P4: X11
        [[1, 1], [0, 0]], # P5: X00 + X01
        [[-1, 0], [1, 0]],# P6: X10 - X00
        [[0, 1], [0, -1]] # P7: X01 - X11
    ]
    
    Y_coeffs = [
        [[1, 0], [0, 1]], # P1: Y00 + Y11
        [[1, 0], [0, 0]], # P2: Y00
        [[0, 1], [0, -1]],# P3: Y01 - Y11
        [[-1, 0], [1, 0]],# P4: Y10 - Y00
        [[0, 0], [0, 1]], # P5: Y11
        [[1, 1], [0, 0]], # P6: Y00 + Y01
        [[0, 1], [0, 1]]  # P7: Y10 + Y11
    ]
    
    Z_coeffs = [
        [[1, 0], [0, 1]], # P1: Z00+1, Z11+1
        [[0, 0], [1, -1]],# P2: Z10+1, Z11-1
        [[0, 1], [0, 1]], # P3: Z01+1, Z11+1
        [[1, 0], [1, 0]], # P4: Z00+1, Z10+1
        [[-1, 1], [0, 0]],# P5: Z00-1, Z01+1
        [[0, 0], [0, 1]], # P6: Z11+1
        [[1, 0], [0, 0]]  # P7: Z00+1
    ]
    
    col = 0
    # Iterate over block indices
    for I in range(3):
        for K in range(3):
            for J in range(3):
                for r in range(7):
                    # Check deadline
                    if time.time() > deadline:
                        break
                    if col >= rank:
                        break
                        
                    # Update U
                    for u in range(2):
                        for x in range(2):
                            coeff = X_coeffs[r][u][x]
                            if coeff != 0:
                                i = 2 * I + u
                                k = 2 * K + x
                                u_idx = i * m + k
                                U[u_idx][col] += coeff
                            
                    # Update V
                    for x in range(2):
                        for y in range(2):
                            coeff = Y_coeffs[r][x][y]
                            if coeff != 0:
                                k = 2 * K + x
                                j = 2 * J + y
                                v_idx = k * p + j
                                V[v_idx][col] += coeff
                                
                    # Update W
                    for u in range(2):
                        for y in range(2):
                            coeff = Z_coeffs[r][u][y]
                            if coeff != 0:
                                i = 2 * I + u
                                j = 2 * J + y
                                w_idx = j * n + i
                                W[w_idx][col] += coeff
                                
                    col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            if col >= rank:
                break
        if col >= rank:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
