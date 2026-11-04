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
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Pre-allocate U, V, W? No, we don't know rank yet.
    # We can calculate rank first.
    # Or just append columns.
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx):
        # u_idx: (i, j) local
        # v_idx: (j, k) local
        # w_idx: (k, i) local
        # This is for a specific block.
        pass
        
    # We will iterate through blocks
    # Block (a, b, c) where a is I block, b is K block, c is J block
    # A is I_a x K_b
    # B is K_b x J_c
    # C is I_a x J_c
    
    for a in range(2):
        for b in range(2):
            for c in range(2):
                n_a = I_sizes[a]
                m_b = K_sizes[b]
                p_c = J_sizes[c]
                
                # Schoolbook rank for this block
                block_rank = n_a * m_b * p_c
                
                for i_local in range(n_a):
                    for j_local in range(m_b):
                        for k_local in range(p_c):
                            # Global indices
                            i_global = I_offsets[a] + i_local
                            j_global = K_offsets[b] + j_local
                            k_global = J_offsets[c] + k_local
                            
                            # U index: i * m + j
                            u_idx = i_global * 5 + j_global
                            # V index: j * p + k
                            v_idx = j_global * 5 + k_global
                            # W index: k * n + i
                            w_idx = k_global * 5 + i_global
                            
                            # Add to U, V, W
                            if not U:
                                U = [[0 for _ in range(121)] for _ in range(25)]
                                V = [[0 for _ in range(121)] for _ in range(25)]
                                W = [[0 for _ in range(121)] for _ in range(25)]
                            
                            col = rank
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                            rank += 1
                            
    return {"U": U, "V": V, "W": W}
