"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Partition: j into 2 groups of 2, k into 4 groups of 2
    # Each sub-problem: 2x2 * 2x2 = 2x2, rank 7 (Strassen)
    # Total: 2 * 4 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    t = 0
    for a in range(2):  # j-group
        for b in range(4):  # k-group
            for s in range(7):  # Strassen step
                # U contributions (A[:, 2a:2a+2])
                if s == 0:  # X00 + X11
                    U[0 * m + 2 * a][t] += 1
                    U[1 * m + 2 * a + 1][t] += 1
                elif s == 1:  # X10 + X11
                    U[1 * m + 2 * a][t] += 1
                    U[1 * m + 2 * a + 1][t] += 1
                elif s == 2:  # X00
                    U[0 * m + 2 * a][t] += 1
                elif s == 3:  # X11
                    U[1 * m + 2 * a + 1][t] += 1
                elif s == 4:  # X00 + X01
                    U[0 * m + 2 * a][t] += 1
                    U[0 * m + 2 * a + 1][t] += 1
                elif s == 5:  # X10 - X00
                    U[1 * m + 2 * a][t] += 1
                    U[0 * m + 2 * a][t] += -1
                elif s == 6:  # X01 - X11
                    U[0 * m + 2 * a + 1][t] += 1
                    U[1 * m + 2 * a + 1][t] += -1
                
                # V contributions (B[2a:2a+2, 2b:2b+2])
                if s == 0:  # Y00 + Y11
                    V[2 * a * p + 2 * b][t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1][t] += 1
                elif s == 1:  # Y00
                    V[2 * a * p + 2 * b][t] += 1
                elif s == 2:  # Y01 - Y11
                    V[2 * a * p + 2 * b + 1][t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1][t] += -1
                elif s == 3:  # Y10 - Y00
                    V[(2 * a + 1) * p + 2 * b][t] += 1
                    V[2 * a * p + 2 * b][t] += -1
                elif s == 4:  # Y11
                    V[(2 * a + 1) * p + 2 * b + 1][t] += 1
                elif s == 5:  # Y00 + Y01
                    V[2 * a * p + 2 * b][t] += 1
                    V[2 * a * p + 2 * b + 1][t] += 1
                elif s == 6:  # Y10 + Y11
                    V[(2 * a + 1) * p + 2 * b][t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1][t] += 1
                
                # W contributions (C[:, 2b:2b+2])
                # Z00 -> C[0, 2b], Z01 -> C[0, 2b+1], Z10 -> C[1, 2b], Z11 -> C[1, 2b+1]
                if s == 0:  # M1: Z00+=1, Z11+=1
                    W[2 * b * n + 0][t] += 1
                    W[(2 * b + 1) * n + 1][t] += 1
                elif s == 1:  # M2: Z10+=1, Z11-=1
                    W[2 * b * n + 1][t] += 1
                    W[(2 * b + 1) * n + 1][t] += -1
                elif s == 2:  # M3: Z01+=1, Z11+=1
                    W[(2 * b + 1) * n + 0][t] += 1
                    W[(2 * b + 1) * n + 1][t] += 1
                elif s == 3:  # M4: Z00+=1, Z10+=1
                    W[2 * b * n + 0][t] += 1
                    W[2 * b * n + 1][t] += 1
                elif s == 4:  # M5: Z00-=1, Z01+=1
                    W[2 * b * n + 0][t] += -1
                    W[(2 * b + 1) * n + 0][t] += 1
                elif s == 5:  # M6: Z11+=1
                    W[(2 * b + 1) * n + 1][t] += 1
                elif s == 6:  # M7: Z00+=1
                    W[2 * b * n + 0][t] += 1
                
                t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
