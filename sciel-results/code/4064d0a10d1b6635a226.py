# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    
    # The directive suggests a Kronecker/block approach.
    # We model the multiplication as 3 independent blocks of 2x3 * 3x3.
    # This results in a rank of 3 * (2*3*3) = 54.
    # We construct the factors explicitly for each block.
    
    num_blocks = p // 3
    p_block = 3
    
    # Total rank
    R = num_blocks * n * m * p_block
    
    # Initialize U, V, W as lists of lists (rows)
    # U: (n*m) x R
    # V: (m*p) x R
    # W: (p*n) x R
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    r_idx = 0
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    # Global indices
                    u_idx = i * m + j
                    v_idx = j * p + (b * p_block + k)
                    w_idx = (b * p_block + k) * n + i
                    
                    U[u_idx][r_idx] = 1
                    V[v_idx][r_idx] = 1
                    W[w_idx][r_idx] = 1
                    
                    r_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
