import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for idx, coeff in u_vals.items():
            u_col[idx] = coeff
        for idx, coeff in v_vals.items():
            v_col[idx] = coeff
        for idx, coeff in w_vals.items():
            w_col[idx] = coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Part 1: A * B_right (4x4 * 4x2) using schoolbook
    for i in range(4):
        for j in range(4):
            for k_prime in range(2):
                k = 4 + k_prime
                u_vals = {i * 4 + j: 1.0}
                v_vals = {j * 6 + k: 1.0}
                w_vals = {k * 4 + i: 1.0}
                add_term(u_vals, v_vals, w_vals)

    # Part 2: A * B_left (4x4 * 4x4) using Strassen
    def a_idx(i, j): return i * 4 + j
    def b_idx(j, k): return j * 6 + k
    def c_idx(i, k): return k * 4 + i
    
    X1 = [
        [{a_idx(0,0): 1.0, a_idx(2,2): 1.0}, {a_idx(0,1): 1.0, a_idx(2,3): 1.0}],
        [{a_idx(1,0): 1.0, a_idx(3,2): 1.0}, {a_idx(1,1): 1.0, a_idx(3,3): 1.0}]
    ]
    X2 = [
        [{a_idx(2,0): 1.0}, {a_idx(2,1): 1.0}],
        [{a_idx(2,2): 1.0, a_idx(3,2): 1.0}, {a_idx(2,3): 1.0, a_idx(3,3): 1.0}]
    ]
    X3 = [
        [{a_idx(0,0): 1.0}, {a_idx(0,1): 1.0}],
        [{a_idx(1,0): 1.0}, {a_idx(1,1): 1.0}]
    ]
    X4 = [
        [{a_idx(2,2): 1.0}, {a_idx(2,3): 1.0}],
        [{a_idx(3,2): 1.0}, {a_idx(3,3): 1.0}]
    ]
    X5 = [
        [{a_idx(0,0): 1.0, a_idx(0,2): 1.0}, {a_idx(0,1): 1.0, a_idx(0,3): 1.0}],
        [{a_idx(1,0): 1.0, a_idx(1,2): 1.0}, {a_idx(1,1): 1.0, a_idx(1,3): 1.0}]
    ]
    X6 = [
        [{a_idx(2,0): 1.0, a_idx(0,0): -1.0}, {a_idx(2,1): 1.0, a_idx(0,1): -1.0}],
        [{a_idx(3,0): 1.0, a_idx(1,0): -1.0}, {a_idx(3,1): 1.0, a_idx(1,1): -1.0}]
    ]
    X7 = [
        [{a_idx(0,2): 1.0, a_idx(2,2): -1.0}, {a_idx(0,3): 1.0, a_idx(2,3): -1.0}],
        [{a_idx(1,2): 1.0, a_idx(3,2): -1.0}, {a_idx(1,3): 1.0, a_idx(3,3): -1.0}]
    ]
    
    Y1 = [
        [{b_idx(0,0): 1.0, b_idx(2,0): 1.0}, {b_idx(0,1): 1.0, b_idx(2,1): 1.0}],
        [{b_idx(1,0): 1.0, b_idx(3,0): 1.0}, {b_idx(1,1): 1.0, b_idx(3,1): 1.0}]
    ]
    Y2 = [
        [{b_idx(0,0): 1.0}, {b_idx(0,1): 1.0}],
        [{b_idx(1,0): 1.0}, {b_idx(1,1): 1.0}]
    ]
    Y3 = [
        [{b_idx(0,2): 1.0, b_idx(2,0): -1.0}, {b_idx(0,3): 1.0, b_idx(2,1): -1.0}],
        [{b_idx(1,2): 1.0, b_idx(3,0): -1.0}, {b_idx(1,3): 1.0, b_idx(3,1): -1.0}]
    ]
    Y4 = [
        [{b_idx(2,0): 1.0, b_idx(0,0): -1.0}, {b_idx(2,1): 1.0, b_idx(0,1): -1.0}],
        [{b_idx(3,0): 1.0, b_idx(1,0): -1.0}, {b_idx(3,1): 1.0, b_idx(1,1): -1.0}]
    ]
    Y5 = [
        [{b_idx(2,0): 1.0}, {b_idx(2,1): 1.0}],
        [{b_idx(3,0): 1.0}, {b_idx(3,1): 1.0}]
    ]
    Y6 = [
        [{b_idx(0,0): 1.0, b_idx(0,2): 1.0}, {b_idx(0,1): 1.0, b_idx(0,3): 1.0}],
        [{b_idx(1,0): 1.0, b_idx(1,2): 1.0}, {b_idx(1,1): 1.0, b_idx(1,3): 1.0}]
    ]
    Y7 = [
        [{b_idx(2,0): 1.0, b_idx(2,2): 1.0}, {b_idx(2,1): 1.0, b_idx(2,3): 1.0}],
        [{b_idx(3,0): 1.0, b_idx(3,2): 1.0}, {b_idx(3,1): 1.0, b_idx(3,3): 1.0}]
    ]
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    M_coeffs = {
        1: {11: 1.0, 22: 1.0},
        2: {21: 1.0, 22: -1.0},
        3: {12: 1.0, 22: 1.0},
        4: {11: 1.0, 21: 1.0},
        5: {11: -1.0, 12: 1.0},
        6: {22: 1.0},
        7: {11: 1.0}
    }
    
    def get_block_offset(block_id, p, q):
        if block_id == 11:
            return (p, q)
        elif block_id == 12:
            return (p, 2 + q)
        elif block_id == 21:
            return (2 + p, q)
        elif block_id == 22:
            return (2 + p, 2 + q)
            
    for l in range(7):
        X_l = Xs[l]
        Y_l = Ys[l]
        for p in range(2):
            for q in range(2):
                for u in range(2):
                    u_vals = X_l[p][u]
                    v_vals = Y_l[u][q]
                    w_vals = {}
                    for block_id, coeff in M_coeffs[l+1].items():
                        i, k = get_block_offset(block_id, p, q)
                        c_idx_val = c_idx(i, k)
                        w_vals[c_idx_val] = w_vals.get(c_idx_val, 0.0) + coeff
                    add_term(u_vals, v_vals, w_vals)
    
    U = [[U_cols[r][i] for r in range(len(U_cols))] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(len(V_cols))] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(len(W_cols))] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
