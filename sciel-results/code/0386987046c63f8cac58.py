"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 multiplication
    # r=0..6 correspond to M1..M7
    # alpha[r][u][v] for A block
    # beta[r][u][v] for B block
    # gamma[r][u][v] for C block
    # u, v in {0, 1}
    
    alpha = [
        [[1, 0], [0, 1]],  # M1: X00 + X11
        [[0, 0], [1, 1]],  # M2: X10 + X11
        [[1, 0], [0, 0]],  # M3: X00
        [[0, 0], [0, 1]],  # M4: X11
        [[1, 1], [0, 0]],  # M5: X00 + X01
        [[-1, 0], [1, 0]], # M6: -X00 + X10
        [[0, 1], [0, -1]], # M7: X01 - X11
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # M1: Y00 + Y11
        [[1, 0], [0, 0]],  # M2: Y00
        [[0, 1], [0, -1]], # M3: Y01 - Y11
        [[-1, 0], [1, 0]], # M4: -Y00 + Y10
        [[0, 0], [0, 1]],  # M5: Y11
        [[1, 1], [0, 0]],  # M6: Y00 + Y01
        [[0, 1], [0, 1]],  # M7: Y10 + Y11
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # M1: C00 + C11
        [[0, 0], [1, -1]], # M2: C10 - C11
        [[0, 1], [0, 1]],  # M3: C01 + C11
        [[1, 0], [1, 0]],  # M4: C00 + C10
        [[-1, 0], [0, 1]], # M5: -C00 + C01
        [[0, 0], [0, 1]],  # M6: C11
        [[1, 0], [0, 0]],  # M7: C00
    ]
    
    # Rank is 9 (block positions) * 7 (Strassen steps) = 63
    rank = 63
    num_blocks = 3 # 3x3 blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for r in range(7):
                    # Fill U
                    for u in range(2):
                        for v in range(2):
                            i = 2 * I + u
                            j = 2 * K + v
                            U[i * m + j][col] = alpha[r][u][v]
                    
                    # Fill V
                    for u in range(2):
                        for v in range(2):
                            j = 2 * K + u
                            k = 2 * J + v
                            V[j * p + k][col] = beta[r][u][v]
                    
                    # Fill W
                    for u in range(2):
                        for v in range(2):
                            k = 2 * J + v
                            i = 2 * I + u
                            W[k * n + i][col] = gamma[r][u][v]
                            
                    col += 1
                    
                    if time.time() > deadline:
                        # If we exceed deadline, we might have incomplete matrices.
                        # But since we are filling sequentially, and the deadline is 145s,
                        # and this loop is very small (63 iterations), it should finish instantly.
                        # This guard is just for safety.
                        pass
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
