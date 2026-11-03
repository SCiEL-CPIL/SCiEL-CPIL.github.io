"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Rank 56 solution using Strassen decomposition on 2x2 blocks
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    t = 0
    for a in range(2):  # j-group: 0 or 1, covering j=2a, 2a+1
        for b in range(4):  # k-group: 0..3, covering k=2b, 2b+1
            for s in range(7):  # Strassen step
                # U contributions (A[:, 2a:2a+2])
                if s == 0:  # X00 + X11
                    U[0 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 1:  # X10 + X11
                    U[1 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 2:  # X00
                    U[0 * m + 2 * a, t] += 1
                elif s == 3:  # X11
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 4:  # X00 + X01
                    U[0 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a + 1, t] += 1
                elif s == 5:  # X10 - X00
                    U[1 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a, t] += -1
                elif s == 6:  # X01 - X11
                    U[0 * m + 2 * a + 1, t] += 1
                    U[1 * m + 2 * a + 1, t] += -1
                
                # V contributions (B[2a:2a+2, 2b:2b+2])
                if s == 0:  # Y00 + Y11
                    V[2 * a * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                elif s == 1:  # Y00
                    V[2 * a * p + 2 * b, t] += 1
                elif s == 2:  # Y01 - Y11
                    V[2 * a * p + 2 * b + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += -1
                elif s == 3:  # Y10 - Y00
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b, t] += -1
                elif s == 4:  # Y11
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                elif s == 5:  # Y00 + Y01
                    V[2 * a * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b + 1, t] += 1
                elif s == 6:  # Y10 + Y11
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                
                # W contributions (C[:, 2b:2b+2])
                if s == 0:  # M1: Z00+=1, Z11+=1
                    W[2 * b * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 1:  # M2: Z10+=1, Z11-=1
                    W[2 * b * n + 1, t] += 1
                    W[(2 * b + 1) * n + 1, t] += -1
                elif s == 2:  # M3: Z01+=1, Z11+=1
                    W[(2 * b + 1) * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 3:  # M4: Z00+=1, Z10+=1
                    W[2 * b * n + 0, t] += 1
                    W[2 * b * n + 1, t] += 1
                elif s == 4:  # M5: Z00-=1, Z01+=1
                    W[2 * b * n + 0, t] += -1
                    W[(2 * b + 1) * n + 0, t] += 1
                elif s == 5:  # M6: Z11+=1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 6:  # M7: Z00+=1
                    W[2 * b * n + 0, t] += 1
                
                t += 1
                
                if time.time() - start_time > deadline:
                    break
            if time.time() - start_time > deadline:
                break
        if time.time() - start_time > deadline:
            break
    if time.time() - start_time > deadline:
        break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
