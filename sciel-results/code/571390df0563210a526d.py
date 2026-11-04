import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: 6x6 is 2x2 blocks of 3x3 matrices.
    # Outer: 2x2 matrix multiply using Strassen (Rank 7).
    # Inner: 3x3 matrix multiply using Laderman (Rank 23).
    # Total Rank = 7 * 23 = 161.
    
    block_size = 3
    dim = n # 6
    
    rank_inner = 23 # Laderman rank
    rank_outer = 7 # Strassen rank
    total_rank = rank_inner * rank_outer
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    
    # Strassen steps for 2x2 block matrix multiplication
    steps_outer = [
        {
            'A': { (0,0): 1, (1,1): 1 },
            'B': { (0,0): 1, (1,1): 1 },
            'C': { (0,0): 1, (1,1): 1 }
        },
        {
            'A': { (1,0): 1, (1,1): 1 },
            'B': { (0,0): 1 },
            'C': { (1,0): 1, (1,1): -1 }
        },
        {
            'A': { (0,0): 1 },
            'B': { (0,1): 1, (1,1): -1 },
            'C': { (0,1): 1, (1,1): 1 }
        },
        {
            'A': { (1,1): 1 },
            'B': { (1,0): 1, (0,0): -1 },
            'C': { (0,0): 1, (1,0): 1 }
        },
        {
            'A': { (0,0): 1, (0,1): 1 },
            'B': { (1,1): 1 },
            'C': { (0,0): -1, (0,1): 1 }
        },
        {
            'A': { (1,0): 1, (0,0): -1 },
            'B': { (0,0): 1, (0,1): 1 },
            'C': { (1,1): 1 }
        },
        {
            'A': { (0,1): 1, (1,1): -1 },
            'B': { (1,0): 1, (1,1): 1 },
            'C': { (0,0): 1 }
        }
    ]
    
    # Laderman steps for 3x3 matrix multiplication
    steps_inner = [
        {'A': {(0,0): 1}, 'B': {(0,0): 1, (1,1): -1}, 'C': {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,1): 1, (1,1): -1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (2,2): 1}, 'B': {(0,0): 1, (2,2): 1}, 'C': {(0,0): 1, (2,2): 1}},
        {'A': {(2,0): 1, (0,0): -1, (2,2): 1}, 'B': {(0,0): 1, (0,2): 1}, 'C': {(0,2): 1, (2,0): 1}},
        {'A': {(0,2): 1, (2,2): 1, (0,0): -1}, 'B': {(2,0): 1, (0,0): 1}, 'C': {(2,0): 1, (0,2): 1}},
        {'A': {(2,2): 1, (0,2): -1}, 'B': {(2,0): 1, (2,2): 1}, 'C': {(2,2): 1}},
        {'A': {(0,0): 1, (2,2): -1}, 'B': {(2,2): 1, (0,2): -1}, 'C': {(2,2): 1}},
        {'A': {(1,0): 1, (1,1): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,0): 1, (1,1): 1}},
        {'A': {(0,1): 1, (1,0): -1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,0): 1, (2,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): 1}, 'B': {(1,0): 1, (2,0): -1}, 'C': {(0,1): 1}},
        {'A': {(1,1): 1, (2,1): 1}, 'B': {(0,0): 1, (1,0): -1}, 'C': {(2,1): 1}},
        {'A': {(0,0): 1, (1,0): 1, (2,0): -1}, 'B': {(1,0): 1, (2,0): 1}, 'C': {(0,0): 1}},
        {'A': {(0,2): 1, (1,2): 1, (2,2): -1}, 'B': {(2,0): 1, (2,2): 1}, 'C': {(0,2): 1}},
        {'A': {(2,2): 1, (0,2): -1, (1,2): -1}, 'B': {(0,2): 1, (1,2): 1}, 'C': {(2,2): 1}},
        {'A': {(0,2): 1, (2,2): 1}, 'B': {(1,2): 1, (0,2): -1}, 'C': {(2,2): 1}},
        {'A': {(1,2): 1, (2,2): 1}, 'B': {(2,1): 1, (1,1): -1}, 'C': {(2,2): 1}},
        {'A': {(1,1): 1, (2,1): 1}, 'B': {(1,2): 1, (0,2): -1}, 'C': {(2,2): 1}},
        {'A': {(0,2): 1, (1,2): 1, (2,2): -1}, 'B': {(0,2): 1, (1,2): 1}, 'C': {(2,2): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,0): 1, (1,1): 1}},
        {'A': {(0,1): 1, (1,1): 1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,1): 1, (1,1): 1}}
    ]
    
    for step_o in steps_outer:
        A_o = step_o['A']
        B_o = step_o['B']
        C_o = step_o['C']
        
        for step_i in steps_inner:
            A_i = step_i['A']
            B_i = step_i['B']
            C_i = step_i['C']
            
            # Update U
            for (I, J), ca_o in A_o.items():
                if ca_o == 0: continue
                for (u, v), ca_i in A_i.items():
                    if ca_i == 0: continue
                    coeff = ca_o * ca_i
                    i_global = I * block_size + u
                    j_global = J * block_size + v
                    idx_u = i_global * dim + j_global
                    U[idx_u][col] += coeff
            
            # Update V
            for (J, K), cb_o in B_o.items():
                if cb_o == 0: continue
                for (v, w), cb_i in B_i.items():
                    if cb_i == 0: continue
                    coeff = cb_o * cb_i
                    j_global = J * block_size + v
                    k_global = K * block_size + w
                    idx_v = j_global * dim + k_global
                    V[idx_v][col] += coeff
            
            # Update W
            for (I, K), cc_o in C_o.items():
                if cc_o == 0: continue
                for (u, w), cc_i in C_i.items():
                    if cc_i == 0: continue
                    coeff = cc_o * cc_i
                    i_global = I * block_size + u
                    k_global = K * block_size + w
                    idx_w = k_global * dim + i_global
                    W[idx_w][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
