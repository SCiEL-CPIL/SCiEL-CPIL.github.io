import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals, col):
        for idx, val in zip(u_indices, u_vals):
            U[idx, col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, col] += val
            
    # Strassen coefficients for 2x2 x 2x2
    # Local indices: A: 00,01,10,11 -> 0,1,2,3
    #               B: 00,01,10,11 -> 0,1,2,3
    #               C: 00,01,10,11 -> 0,1,2,3
    strassen_terms = [
        # (u_idx, u_val), (v_idx, v_val), (w_idx, w_val)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (1, 1), (2, 1), (3, -1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]), # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]), # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]), # M5
        ([(0, -1), (2, 1)], [(0, 1), (1, 1)], [(3, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]), # M7
    ]
    
    # Block 1a: A(0:2, 0:2), B(0:2, 0:2), C(0:2, 0:2)
    # A global: 0,1,3,4
    # B global: 0,1,4,5
    # C global: 0,1,2,3
    a_map_1a = [0, 1, 3, 4]
    b_map_1a = [0, 1, 4, 5]
    c_map_1a = [0, 1, 2, 3]
    
    for t_idx in range(7):
        u_data, v_data, w_data = strassen_terms[t_idx]
        u_indices = [a_map_1a[i] for i, _ in u_data]
        u_vals = [v for _, v in u_data]
        v_indices = [b_map_1a[i] for i, _ in v_data]
        v_vals = [v for _, v in v_data]
        w_indices = [c_map_1a[i] for i, _ in w_data]
        w_vals = [v for _, v in w_data]
        add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals, t_idx)
        
    # Block 1b: A(0:2, 0:2), B(0:2, 2:4), C(0:2, 2:4)
    # A global: 0,1,3,4
    # B global: 2,3,6,7
    # C global: 4,5,6,7
    a_map_1b = [0, 1, 3, 4]
    b_map_1b = [2, 3, 6, 7]
    c_map_1b = [4, 5, 6, 7]
    
    for t_idx in range(7):
        u_data, v_data, w_data = strassen_terms[t_idx]
        u_indices = [a_map_1b[i] for i, _ in u_data]
        u_vals = [v for _, v in u_data]
        v_indices = [b_map_1b[i] for i, _ in v_data]
        v_vals = [v for _, v in v_data]
        w_indices = [c_map_1b[i] for i, _ in w_data]
        w_vals = [v for _, v in w_data]
        add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals, 7 + t_idx)
        
    # Block 2: A(0:2, 2:3), B(2:3, 0:4), C(0:2, 0:4)
    # A global: 2, 5
    # B global: 8, 9, 10, 11
    # C global: 0, 1, 4, 5, 2, 3, 6, 7
    
    naive_terms = [
        (2, 8, 0), # A02 B20 C00
        (2, 9, 1), # A02 B21 C01
        (2, 10, 4), # A02 B22 C02
        (2, 11, 5), # A02 B23 C03
        (5, 8, 2), # A12 B20 C10
        (5, 9, 3), # A12 B21 C11
        (5, 10, 6), # A12 B22 C12
        (5, 11, 7), # A12 B23 C13
    ]
    
    for i, (u_idx, v_idx, w_idx) in enumerate(naive_terms):
        add_term([u_idx], [1], [v_idx], [1], [w_idx], [1], 14 + i)
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
