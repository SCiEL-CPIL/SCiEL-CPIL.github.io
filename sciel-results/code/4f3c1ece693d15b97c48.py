import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 base decomposition
    # Indices: 0..3 for 2x2 blocks
    # U_2, V_2, W_2 are 4x7
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1]
    ], dtype=np.float64)
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ], dtype=np.float64)
    
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)

    # Specs for 4x4x4 Strassen
    # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    A_specs = [
        [(1, 0), (1, 3)], 
        [(1, 2), (1, 3)], 
        [(1, 0)],         
        [(1, 3)],         
        [(1, 0), (1, 1)], 
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)]
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], 
        [(1, 0)],         
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],         
        [(1, 0), (1, 1)], 
        [(1, 2), (1, 3)]
    ]
    
    C_specs = [
        [(1, 0), (1, 3)], 
        [(1, 2), (-1, 3)], 
        [(1, 1), (1, 3)], 
        [(1, 0), (1, 2)], 
        [(-1, 0), (1, 1)], 
        [(1, 3)], 
        [(1, 0)]
    ]
    
    # Precompute block mappings for efficiency and distinct structure
    # block_idx -> (bi, bj)
    block_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    
    # Build the decomposition
    # We have 2 halves of the 8 columns (each 4 wide)
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                # Update U
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = block_map[block_idx]
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            ga = gr * 4 + gc
                            a_eff = r * 2 + c
                            U[ga, col] += coeff_a * U_2[a_eff, m_s]
                
                # Update V
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = block_map[block_idx]
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + half * 4
                            gb = gr * 8 + gk
                            b_eff = r * 2 + c
                            V[gb, col] += coeff_b * V_2[b_eff, m_s]
                
                # Update W
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = block_map[c_block_idx]
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + half * 4
                            gw = gk * 4 + gi
                            p_eff = r * 2 + c
                            W[gw, col] += coeff_c * W_2[p_eff, m_s]
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
