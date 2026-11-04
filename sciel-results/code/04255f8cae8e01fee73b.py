"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decompose dimensions into blocks:
    # n=3: [2, 1]
    # m=5: [2, 3]
    # p=6: [2, 4]
    
    n_blocks = [2, 1]
    m_blocks = [2, 3]
    p_blocks = [2, 4]
    
    # Calculate total rank as sum of products of block sizes
    rank = sum(ni * mi * pi for ni in n_blocks for mi in m_blocks for pi in p_blocks)
    
    # Build U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for ni_idx, ni in enumerate(n_blocks):
        n_start = sum(n_blocks[:ni_idx])
        for mi_idx, mi in enumerate(m_blocks):
            m_start = sum(m_blocks[:mi_idx])
            for pi_idx, pi in enumerate(p_blocks):
                p_start = sum(p_blocks[:pi_idx])
                
                # For each element in this block multiplication
                for i in range(ni):
                    for j in range(mi):
                        for k in range(pi):
                            # U index: (n_start + i) * m + (m_start + j)
                            u_idx = (n_start + i) * m + (m_start + j)
                            # V index: (m_start + j) * p + (p_start + k)
                            v_idx = (m_start + j) * p + (p_start + k)
                            # W index: (p_start + k) * n + (n_start + i)
                            w_idx = (p_start + k) * n + (n_start + i)
                            
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
