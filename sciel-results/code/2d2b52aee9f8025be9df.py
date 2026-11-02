import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen 2x2 decomposition coefficients
    # U_loc[s][u][v], V_loc[s][v][w], W_loc[s][w][u]
    # s in 0..6
    U_loc = np.zeros((7, 2, 2), dtype=int)
    V_loc = np.zeros((7, 2, 2), dtype=int)
    W_loc = np.zeros((7, 2, 2), dtype=int)
    
    # s=0: m1 = (a+d)(e+h)
    U_loc[0, 0, 0] = 1; U_loc[0, 1, 1] = 1
    V_loc[0, 0, 0] = 1; V_loc[0, 1, 1] = 1
    W_loc[0, 0, 0] = 1; W_loc[0, 1, 1] = 1
    
    # s=1: m2 = (c+d)e
    U_loc[1, 1, 0] = 1; U_loc[1, 1, 1] = 1
    V_loc[1, 0, 0] = 1
    W_loc[1, 0, 1] = 1; W_loc[1, 1, 1] = -1
    
    # s=2: m3 = a(f-h)
    U_loc[2, 0, 0] = 1
    V_loc[2, 1, 0] = 1; V_loc[2, 0, 0] = -1
    W_loc[2, 1, 0] = 1; W_loc[2, 1, 1] = 1
    
    # s=3: m4 = d(g-e)
    U_loc[3, 1, 1] = 1
    V_loc[3, 0, 1] = 1; V_loc[3, 0, 0] = -1
    W_loc[3, 0, 0] = 1; W_loc[3, 0, 1] = 1
    
    # s=4: m5 = (a+b)h
    U_loc[4, 0, 0] = 1; U_loc[4, 0, 1] = 1
    V_loc[4, 1, 1] = 1
    W_loc[4, 0, 0] = -1; W_loc[4, 1, 0] = 1
    
    # s=5: m6 = (c-a)(e+f)
    U_loc[5, 1, 0] = 1; U_loc[5, 0, 0] = -1
    V_loc[5, 0, 0] = 1; V_loc[5, 0, 1] = 1
    W_loc[5, 1, 1] = 1
    
    # s=6: m7 = (b-d)(g+h)
    U_loc[6, 0, 1] = 1; U_loc[6, 1, 1] = -1
    V_loc[6, 1, 0] = 1; V_loc[6, 1, 1] = 1
    W_loc[6, 0, 0] = 1
    
    rank = (num_blocks * num_blocks) * num_blocks * 7
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for I in range(num_blocks):
        for K in range(num_blocks):
            for J in range(num_blocks):
                for s in range(7):
                    for u in range(block_size):
                        for v in range(block_size):
                            i = I * block_size + u
                            j = J * block_size + v
                            U[i * m + j, r] = U_loc[s, u, v]
                            
                            for w in range(block_size):
                                k = K * block_size + w
                                V[j * p + k, r] = V_loc[s, v, w]
                                W[k * n + i, r] = W_loc[s, w, u]
                    r += 1
                    
    if time.time() > deadline:
        # Should not happen for this small size
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
