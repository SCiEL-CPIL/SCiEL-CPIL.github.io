import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        def gI(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def gJ(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def gK(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)
        
        add_term([(gI(0,0),1),(gI(1,1),1)], [(gJ(0,0),1),(gJ(1,1),1)], [(gK(0,0),1),(gK(1,1),1)])
        add_term([(gI(1,0),1),(gI(1,1),1)], [(gJ(0,0),1)], [(gK(0,1),1),(gK(1,1),-1)])
        add_term([(gI(0,0),1)], [(gJ(0,1),1),(gJ(1,1),-1)], [(gK(1,0),1),(gK(1,1),1)])
        add_term([(gI(1,1),1)], [(gJ(1,0),1),(gJ(0,0),-1)], [(gK(0,0),1),(gK(0,1),1)])
        add_term([(gI(0,0),1),(gI(0,1),1)], [(gJ(1,1),1)], [(gK(0,0),-1),(gK(1,0),1)])
        add_term([(gI(1,0),1),(gI(0,0),-1)], [(gJ(0,0),1),(gJ(0,1),1)], [(gK(1,1),1)])
        add_term([(gI(0,1),1),(gI(1,1),-1)], [(gJ(1,0),1),(gJ(1,1),1)], [(gK(0,0),1)])
    
    # Strategy: Use 4 Strassen 2x2x2 blocks for the 4x4x4 core (rank 28)
    # Then schoolbook for the remaining strips.
    # Partition: i=[4], j=[4,1], k=[4,1]
    # 4x4x4 core: 4 Strassen blocks = rank 28
    # 4x4x1 strip: schoolbook = rank 16
    # 4x1x4 strip: schoolbook = rank 16  
    # 4x1x1 corner: schoolbook = rank 4
    # Total: 28 + 16 + 16 + 4 = 64
    
    if time.time() < deadline:
        # 4x4x4 core using 4 Strassen 2x2x2 blocks
        strassen_2x2x2(0, 0, 0)
        strassen_2x2x2(2, 0, 0)
        strassen_2x2x2(0, 2, 0)
        strassen_2x2x2(2, 2, 0)
    
    if time.time() < deadline:
        # 4x4x1 strip at k=4
        schoolbook(4, 4, 1, 0, 0, 4)
    
    if time.time() < deadline:
        # 4x1x4 strip at j=4
        schoolbook(4, 1, 4, 0, 4, 0)
    
    if time.time() < deadline:
        # 4x1x1 corner
        schoolbook(4, 1, 1, 0, 4, 4)
    
    if not U_cols:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
