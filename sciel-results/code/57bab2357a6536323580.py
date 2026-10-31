"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen vectors
    u_2x2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_2x2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_2x2 = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for t1 in range(7):
        if time.time() > deadline:
            break
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for i2 in range(2):
                    idx = i1 * 2 + i2
                    U[idx][col] = u_2x2[t1][i1] * u_2x2[t2][i2]
                    V[idx][col] = v_2x2[t1][i1] * v_2x2[t2][i2]
                    W[idx][col] = w_2x2[t1][i1] * w_2x2[t2][i2]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
