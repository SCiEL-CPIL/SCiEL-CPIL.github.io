import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # 1. Strassen for 4x4x4 part (k in 0..3)
    # Define 2x2 Strassen tensors (local indices 0,1)
    # U[i, j], V[j, k], W[k, i]
    strassen_2x2 = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
    ]
    
    # Define 4x4 Strassen block combinations (block indices 0,1)
    # U[I, J], V[J, K], W[K, I]
    strassen_4x4_blocks = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
    ]
    
    col = 0
    for s in range(7):
        U_block, V_block, W_block = strassen_4x4_blocks[s]
        for t in range(7):
            U_elem, V_elem, W_elem = strassen_2x2[t]
            
            # Combine block and element levels
            for (I, J), cU in U_block.items():
                for (i, j), dU in U_elem.items():
                    gi = 2 * I + i
                    gj = 2 * J + j
                    U[gi * m + gj, col] += cU * dU
            
            for (J, K), cV in V_block.items():
                for (j, k), dV in V_elem.items():
                    gj = 2 * J + j
                    gk = 2 * K + k
                    V[gj * p + gk, col] += cV * dV
            
            for (K, I), cW in W_block.items():
                for (k, i), dW in W_elem.items():
                    gk = 2 * K + k
                    gi = 2 * I + i
                    W[gk * n + gi, col] += cW * dW
            
            col += 1
    
    # 2. Schoolbook for 4x4x3 part (k in 4..6)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
