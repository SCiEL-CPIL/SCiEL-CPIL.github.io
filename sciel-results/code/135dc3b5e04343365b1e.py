import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: 4x4 x 4x4 (B columns 0-3) using Strassen tensor product
    # Block-level Strassen steps for 2x2 block matrices
    block_steps = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),                # M2
        ([(0,1)], [(2,1),(3,-1)], [(1,1)]),               # M3
        ([(3,1)], [(1,1),(0,-1)], [(0,1),(2,1)]),         # M4
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),         # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),         # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),         # M7
    ]
    
    # Sub-level Strassen for 2x2x2 scalar multiplications
    sub_steps = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1}),
        ({0:1}, {2:1, 3:-1}, {1:1}),
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    for (a_bks, b_bks, c_bks) in block_steps:
        for (a_sub, b_sub, c_sub) in sub_steps:
            u_col = {}
            v_col = {}
            w_col = {}
            
            for (bi, sign_a) in a_bks:
                row_start = 2 * (bi // 2)
                col_start = 2 * (bi % 2)
                for (li, sa) in a_sub.items():
                    li_row = li // 2
                    li_col = li % 2
                    gr = row_start + li_row
                    gc = col_start + li_col
                    a_idx = gr * 4 + gc
                    u_col[a_idx] = u_col.get(a_idx, 0) + sign_a * sa
            
            for (bi, sign_b) in b_bks:
                row_start = 2 * (bi // 2)
                col_start = 2 * (bi % 2)
                for (li, sb) in b_sub.items():
                    li_row = li // 2
                    li_col = li % 2
                    gr = row_start + li_row
                    gc = col_start + li_col
                    b_idx = gr * 5 + gc
                    v_col[b_idx] = v_col.get(b_idx, 0) + sign_b * sb
            
            for (bi, sign_c) in c_bks:
                row_start = 2 * (bi // 2)
                col_start = 2 * (bi % 2)
                for (li, sc) in c_sub.items():
                    li_row = li // 2
                    li_col = li % 2
                    gr = row_start + li_row
                    gc = col_start + li_col
                    c_idx = gr * 4 + gc
                    w_col[c_idx] = w_col.get(c_idx, 0) + sign_c * sc
            
            u_vec = [0] * 16
            for idx, val in u_col.items():
                u_vec[idx] = val
            v_vec = [0] * 20
            for idx, val in v_col.items():
                v_vec[idx] = val
            w_vec = [0] * 20
            for idx, val in w_col.items():
                w_vec[idx] = val
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
    
    # Part 2: 4x4 x 4x1 (B column 4)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 20
            w_vec = [0] * 20
            u_vec[i*4+j] = 1
            v_vec[j*5+4] = 1
            w_vec[4*4+i] = 1
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
    
    rank = len(U_list)
    
    # Verify
    U_arr = np.array(U_list, dtype=float).T
    V_arr = np.array(V_list, dtype=float).T
    W_arr = np.array(W_list, dtype=float).T
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_target = np.zeros((16, 20, 20))
    for i in range(4):
        for j in range(4):
            for k in range(5):
                a = i*4+j
                b = j*5+k
                c = k*4+i
                T_target[a,b,c] = 1.0
    
    error = np.sum((T - T_target)**2)
    
    if error < 1e-10:
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    else:
        rank_sb = 80
        U_sb = [[0]*rank_sb for _ in range(16)]
        V_sb = [[0]*rank_sb for _ in range(20)]
        W_sb = [[0]*rank_sb for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_sb[i*4+j][col] = 1
                    V_sb[j*5+k][col] = 1
                    W_sb[k*4+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
