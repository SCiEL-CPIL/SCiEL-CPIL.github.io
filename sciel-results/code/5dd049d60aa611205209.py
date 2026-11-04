import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Bottom (1x5x6): 30
    # Top (2x5x6):
    #   Part A (2x2x6): 3 * 7 = 21
    #   Part B (2x3x6): 2 * (4 + 7 + 6) = 34
    # Total Rank = 30 + 21 + 34 = 85
    rank = 85
    
    # U: n*m x rank = 15 x 85
    # V: m*p x rank = 30 x 85
    # W: p*n x rank = 18 x 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for idx, val in u_list:
            U[idx][col] += val
        for idx, val in v_list:
            V[idx][col] += val
        for idx, val in w_list:
            W[idx][col] += val

    def get_u_idx(i, j):
        return i * m + j

    def get_v_idx(j, k):
        return j * p + k

    def get_w_idx(k, i):
        return k * n + i

    col = 0
    
    # --- Bottom Block (1x5x6) ---
    # i=2
    for k in range(p):
        for j in range(m):
            # A[2,j] * B[j,k] -> C[k,2]
            add_term(col, 
                     [(get_u_idx(2, j), 1)], 
                     [(get_v_idx(j, k), 1)], 
                     [(get_w_idx(k, 2), 1)])
            col += 1

    # --- Top Block (2x5x6) ---
    
    # Part A: j in {0,1} (2x2x6)
    # Split p=6 into 3 blocks of 2
    for k_start in [0, 2, 4]:
        # 2x2x2 Strassen block
        # A rows 0,1; A cols 0,1
        # B rows 0,1; B cols k_start, k_start+1
        # C rows 0,1; C cols k_start, k_start+1
        
        # Strassen Coefficients
        # m1 = (A00+A10)(B00+B01)
        # m2 = (A10+A11)(B00)
        # m3 = (A00)(B01-B10)
        # m4 = (A11)(B11-B01)
        # m5 = (A00+A10)(B11)
        # m6 = (A01-A11)(B00+B10)
        # m7 = (A10-A00)(B01+B11)
        
        # m1
        u = [(get_u_idx(0,0), 1), (get_u_idx(1,0), 1)]
        v = [(get_v_idx(0,k_start), 1), (get_v_idx(0,k_start+1), 1)]
        w = [(get_w_idx(k_start,0), 1), (get_w_idx(k_start+1,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m2
        u = [(get_u_idx(1,0), 1), (get_u_idx(1,1), 1)]
        v = [(get_v_idx(0,k_start), 1)]
        w = [(get_w_idx(k_start,1), 1), (get_w_idx(k_start+1,1), -1)]
        add_term(col, u, v, w); col += 1
        
        # m3
        u = [(get_u_idx(0,0), 1)]
        v = [(get_v_idx(0,k_start+1), 1), (get_v_idx(1,k_start), -1)]
        w = [(get_w_idx(k_start+1,0), 1), (get_w_idx(k_start+1,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m4
        u = [(get_u_idx(1,1), 1)]
        v = [(get_v_idx(1,k_start+1), 1), (get_v_idx(0,k_start+1), -1)]
        w = [(get_w_idx(k_start,0), 1), (get_w_idx(k_start,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m5
        u = [(get_u_idx(0,0), 1), (get_u_idx(1,0), 1)]
        v = [(get_v_idx(1,k_start+1), 1)]
        w = [(get_w_idx(k_start,0), -1), (get_w_idx(k_start+1,0), 1)]
        add_term(col, u, v, w); col += 1
        
        # m6
        u = [(get_u_idx(0,1), 1), (get_u_idx(1,1), -1)]
        v = [(get_v_idx(0,k_start), 1), (get_v_idx(1,k_start), 1)]
        w = [(get_w_idx(k_start+1,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m7
        u = [(get_u_idx(1,0), 1), (get_u_idx(0,0), -1)]
        v = [(get_v_idx(0,k_start+1), 1), (get_v_idx(1,k_start+1), 1)]
        w = [(get_w_idx(k_start,0), 1), (get_w_idx(k_start+1,1), 1)]
        add_term(col, u, v, w); col += 1

    # Part B: j in {2,3,4} (2x3x6)
    # Split p=6 into 2 blocks of 3
    for k_start in [0, 3]:
        # 2x3x3 block
        # A rows 0,1; A cols 2,3,4
        # B rows 2,3,4; B cols k_start..k_start+2
        # C rows 0,1; C cols k_start..k_start+2
        
        # Sub B1a: 2x2x1 (j=2,3; k=k_start)
        # Rank 4
        for i in [0, 1]:
            for j_local in [0, 1]:
                j_global = 2 + j_local
                # A[i, j_global] * B[j_global, k_start] -> C[i, k_start]
                add_term(col, 
                         [(get_u_idx(i, j_global), 1)], 
                         [(get_v_idx(j_global, k_start), 1)], 
                         [(get_w_idx(k_start, i), 1)])
                col += 1

        # Sub B1b: 2x2x2 (j=2,3; k=k_start+1, k_start+2)
        # Strassen
        k1 = k_start + 1
        k2 = k_start + 2
        
        # m1
        u = [(get_u_idx(0,2), 1), (get_u_idx(1,2), 1)]
        v = [(get_v_idx(2,k1), 1), (get_v_idx(2,k2), 1)]
        w = [(get_w_idx(k1,0), 1), (get_w_idx(k2,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m2
        u = [(get_u_idx(1,2), 1), (get_u_idx(1,3), 1)]
        v = [(get_v_idx(2,k1), 1)]
        w = [(get_w_idx(k1,1), 1), (get_w_idx(k2,1), -1)]
        add_term(col, u, v, w); col += 1
        
        # m3
        u = [(get_u_idx(0,2), 1)]
        v = [(get_v_idx(2,k2), 1), (get_v_idx(3,k1), -1)]
        w = [(get_w_idx(k2,0), 1), (get_w_idx(k2,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m4
        u = [(get_u_idx(1,3), 1)]
        v = [(get_v_idx(3,k2), 1), (get_v_idx(2,k2), -1)]
        w = [(get_w_idx(k1,0), 1), (get_w_idx(k1,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m5
        u = [(get_u_idx(0,2), 1), (get_u_idx(1,2), 1)]
        v = [(get_v_idx(3,k2), 1)]
        w = [(get_w_idx(k1,0), -1), (get_w_idx(k2,0), 1)]
        add_term(col, u, v, w); col += 1
        
        # m6
        u = [(get_u_idx(0,3), 1), (get_u_idx(1,3), -1)]
        v = [(get_v_idx(2,k1), 1), (get_v_idx(3,k1), 1)]
        w = [(get_w_idx(k2,1), 1)]
        add_term(col, u, v, w); col += 1
        
        # m7
        u = [(get_u_idx(1,2), 1), (get_u_idx(0,2), -1)]
        v = [(get_v_idx(2,k2), 1), (get_v_idx(3,k2), 1)]
        w = [(get_w_idx(k1,0), 1), (get_w_idx(k2,1), 1)]
        add_term(col, u, v, w); col += 1

        # Sub B2: 2x1x3 (j=4; k=k_start..k_start+2)
        # Rank 6
        for i in [0, 1]:
            for k_local in range(3):
                k_global = k_start + k_local
                # A[i, 4] * B[4, k_global] -> C[i, k_global]
                add_term(col, 
                         [(get_u_idx(i, 4), 1)], 
                         [(get_v_idx(4, k_global), 1)], 
                         [(get_w_idx(k_global, i), 1)])
                col += 1

    return {"U": U, "V": V, "W": W}
