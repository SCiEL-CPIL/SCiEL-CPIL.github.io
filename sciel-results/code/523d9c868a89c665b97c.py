"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2x2
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Global dimensions
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    block_size = 2
    num_blocks = n // block_size # 3
    
    rank = (num_blocks ** 3) * 7 # 27 * 7 = 189
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col_idx = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for rs in range(7):
                    # Local indices
                    # u, v, w in 0..1
                    for u in range(block_size):
                        for v in range(block_size):
                            # Global i, j
                            i = I * block_size + u
                            j = J * block_size + v
                            u_idx = u * block_size + v
                            
                            val_u = U_s[u_idx, rs]
                            if val_u != 0:
                                U[i * m + j, col_idx] = val_u
                            
                            for w in range(block_size):
                                # Global k
                                k = K * block_size + w
                                v_idx = v * block_size + w
                                w_idx = w * block_size + u
                                
                                val_v = V_s[v_idx, rs]
                                if val_v != 0:
                                    V[j * p + k, col_idx] = val_v
                                    
                                val_w = W_s[w_idx, rs]
                                if val_w != 0:
                                    W[k * n + i, col_idx] = val_w
                    col_idx += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
