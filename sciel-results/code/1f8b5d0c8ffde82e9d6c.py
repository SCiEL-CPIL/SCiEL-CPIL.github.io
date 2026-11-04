import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Block sizes
    # I: 3, 2
    # K: 3, 2
    # J: 3, 2
    I_sizes = [3, 2]
    K_sizes = [3, 2]
    J_sizes = [3, 2]
    
    # Offsets
    I_offsets = [0, 3]
    K_offsets = [0, 3]
    J_offsets = [0, 3]
    
    # Calculate total rank
    total_rank = 0
    block_specs = []
    
    for a in range(2):
        for b in range(2):
            for c in range(2):
                n_a = I_sizes[a]
                m_b = K_sizes[b]
                p_c = J_sizes[c]
                
                # Schoolbook rank for this block
                block_rank = n_a * m_b * p_c
                total_rank += block_rank
                
                block_specs.append((a, b, c, n_a, m_b, p_c, block_rank))
                
    # Initialize U, V, W
    # U: (n*m, rank) -> (25, total_rank)
    # V: (m*p, rank) -> (25, total_rank)
    # W: (p*n, rank) -> (25, total_rank)
    U = np.zeros((25, total_rank), dtype=float)
    V = np.zeros((25, total_rank), dtype=float)
    W = np.zeros((25, total_rank), dtype=float)
    
    col_idx = 0
    for a, b, c, n_a, m_b, p_c, block_rank in block_specs:
        I_off = I_offsets[a]
        K_off = K_offsets[b]
        J_off = J_offsets[c]
        
        for i_local in range(n_a):
            for j_local in range(m_b):
                for k_local in range(p_c):
                    # Global indices
                    i_global = I_off + i_local
                    j_global = K_off + j_local
                    k_global = J_off + k_local
                    
                    # U index: i * m + j
                    u_idx = i_global * 5 + j_global
                    # V index: j * p + k
                    v_idx = j_global * 5 + k_global
                    # W index: k * n + i
                    w_idx = k_global * 5 + i_global
                    
                    # Add to U, V, W
                    U[u_idx, col_idx] = 1.0
                    V[v_idx, col_idx] = 1.0
                    W[w_idx, col_idx] = 1.0
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
