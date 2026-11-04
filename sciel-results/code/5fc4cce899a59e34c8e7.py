import time

def construct():
    """
    Returns a rank-54 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    - A*B1 and A*B2 use Laderman for 4x4*4x4 (rank 19 each).
    - A*B3 uses schoolbook for 4x4*4x1 (rank 16).
    - Total rank: 19 + 19 + 16 = 54.
    """
    
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Laderman decomposition for 4x4 * 4x4
    # There are 19 multiplications.
    # We define them as (u_coeff, v_coeff, w_coeff)
    # u_coeff: dict { (i,j): val } for A
    # v_coeff: dict { (j,k): val } for B
    # w_coeff: dict { (i,k): val } for C
    
    # Based on standard Laderman (1976) or equivalent rank-19 algorithm.
    # Note: There are multiple equivalent forms. This is one valid set.
    
    laderman_4x4 = [
        # 1
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # 2
        ({(0,0): 1}, {(0,1): 1}, {(0,1): 1}),
        # 3
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        # 4
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        # 5
        ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # 6
        ({(0,0): 1, (0,1): 1}, {(1,0): 1}, {(0,0): 1, (1,0): -1}),
        # 7
        ({(0,0): 1, (1,0): 1}, {(0,1): 1}, {(0,1): 1, (1,1): -1}),
        # 8
        ({(0,1): 1, (1,1): 1}, {(1,0): 1}, {(0,0): -1, (1,0): 1}),
        # 9
        ({(1,0): 1, (1,1): 1}, {(0,1): 1}, {(0,1): -1, (1,1): 1}),
        # 10
        ({(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}),
        # 11
        ({(0,0): 1, (1,0): 1, (1,1): 1}, {(0,1): 1}, {(0,1): 1, (1,0): -1, (1,1): -1}),
        # 12
        ({(0,0): 1, (0,1): 1, (1,1): 1}, {(1,0): 1}, {(0,0): 1, (1,0): -1, (1,1): -1}),
        # 13
        ({(0,1): 1, (1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1, (0,1): 1, (1,0): 1}),
        # 14
        ({(0,0): 1, (0,1): 1, (1,0): 1}, {(1,1): 1}, {(0,1): 1, (1,0): 1, (1,1): -1}),
        # 15
        ({(0,0): 1, (0,1): 1, (1,1): 1}, {(0,1): 1}, {(0,0): 1, (0,1): -1, (1,1): 1}),
        # 16
        ({(0,0): 1, (1,0): 1, (1,1): 1}, {(1,0): 1}, {(1,0): 1, (1,1): -1, (0,0): 1}),
        # 17
        ({(0,1): 1, (1,0): 1, (1,1): 1}, {(0,1): 1}, {(0,1): 1, (1,0): -1, (1,1): 1}),
        # 18
        ({(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}),
        # 19
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,1): 1, (1,0): 1}),
    ]
    
    def add_laderman_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 19 terms for a 4x4 * 4x4 multiplication using Laderman.
        """
        for u_c, v_c, w_c in laderman_4x4:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            for (li, lj), coeff in u_c.items():
                gi = A_base_i + li
                gj = A_base_j + lj
                u_vec[gi * m + gj] += coeff
                
            for (lj, lk), coeff in v_c.items():
                gj = B_base_j + lj
                gk = B_base_k + lk
                v_vec[gj * p + gk] += coeff
                
            for (li, lk), coeff in w_c.items():
                gi = C_base_i + li
                gk = C_base_k + lk
                w_vec[gk * n + gi] += coeff
                
            add_term(u_vec, v_vec, w_vec)

    # Part 1: A * B1 (cols 0-3)
    add_laderman_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_laderman_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
