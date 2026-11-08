import numpy as np

def construct():
    n, m, p = 4, 5, 5
    max_rank = 100
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, rank):
        if rank < max_rank:
            U[u_idx][rank] += val_u
            V[v_idx][rank] += val_v
            W[w_idx][rank] += val_w

    def decompose(n_sub, m_sub, p_sub, 
                  u_row_off, u_col_off, 
                  v_row_off, v_col_off, 
                  w_res_row_off, w_res_col_off,
                  rank_start):
        
        # Base cases
        if n_sub == 1 and m_sub == 1 and p_sub == 1:
            u_lin = u_row_off * m + u_col_off
            v_lin = v_row_off * p + v_col_off
            w_lin = w_res_col_off * n + w_res_row_off
            add_term(u_lin, v_lin, w_lin, 1, 1, 1, rank_start)
            return 1

        if n_sub == 1:
            rank = rank_start
            for j in range(m_sub):
                for k in range(p_sub):
                    u_lin = u_row_off * m + (u_col_off + j)
                    v_lin = (v_row_off + j) * p + (v_col_off + k)
                    w_lin = (w_res_col_off + k) * n + w_res_row_off
                    add_term(u_lin, v_lin, w_lin, 1, 1, 1, rank)
                    rank += 1
            return rank - rank_start

        if p_sub == 1:
            rank = rank_start
            for i in range(n_sub):
                for j in range(m_sub):
                    u_lin = (u_row_off + i) * m + (u_col_off + j)
                    v_lin = (v_row_off + j) * p + v_col_off
                    w_lin = w_res_col_off * n + (w_res_row_off + i)
                    add_term(u_lin, v_lin, w_lin, 1, 1, 1, rank)
                    rank += 1
            return rank - rank_start

        if m_sub == 1:
            rank = rank_start
            for i in range(n_sub):
                for k in range(p_sub):
                    u_lin = (u_row_off + i) * m + u_col_off
                    v_lin = v_row_off * p + (v_col_off + k)
                    w_lin = (w_res_col_off + k) * n + (w_res_row_off + i)
                    add_term(u_lin, v_lin, w_lin, 1, 1, 1, rank)
                    rank += 1
            return rank - rank_start

        # Recursive case
        # Split dimensions
        n1 = n_sub // 2
        n2 = n_sub - n1
        m1 = m_sub // 2
        m2 = m_sub - m1
        p1 = p_sub // 2
        p2 = p_sub - p1
        
        # 8 blocks
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        r = rank_start
        
        # A11 B11 -> C11
        r += decompose(n1, m1, p1, 
                       u_row_off, u_col_off, 
                       v_row_off, v_col_off, 
                       w_res_row_off, w_res_col_off, r)
        
        # A12 B21 -> C11
        r += decompose(n1, m2, p1, 
                       u_row_off, u_col_off + m1, 
                       v_row_off + m1, v_col_off, 
                       w_res_row_off, w_res_col_off, r)
        
        # A11 B12 -> C12
        r += decompose(n1, m1, p2, 
                       u_row_off, u_col_off, 
                       v_row_off, v_col_off + p1, 
                       w_res_row_off, w_res_col_off + p1, r)
        
        # A12 B22 -> C12
        r += decompose(n1, m2, p2, 
                       u_row_off, u_col_off + m1, 
                       v_row_off + m1, v_col_off + p1, 
                       w_res_row_off, w_res_col_off + p1, r)
        
        # A21 B11 -> C21
        r += decompose(n2, m1, p1, 
                       u_row_off + n1, u_col_off, 
                       v_row_off, v_col_off, 
                       w_res_row_off + n1, w_res_col_off, r)
        
        # A22 B21 -> C21
        r += decompose(n2, m2, p1, 
                       u_row_off + n1, u_col_off + m1, 
                       v_row_off + m1, v_col_off, 
                       w_res_row_off + n1, w_res_col_off, r)
        
        # A21 B12 -> C22
        r += decompose(n2, m1, p2, 
                       u_row_off + n1, u_col_off, 
                       v_row_off, v_col_off + p1, 
                       w_res_row_off + n1, w_res_col_off + p1, r)
        
        # A22 B22 -> C22
        r += decompose(n2, m2, p2, 
                       u_row_off + n1, u_col_off + m1, 
                       v_row_off + m1, v_col_off + p1, 
                       w_res_row_off + n1, w_res_col_off + p1, r)
        
        return r - rank_start

    total_rank = decompose(n, m, p, 0, 0, 0, 0, 0, 0, 0)
    
    U = [row[:total_rank] for row in U]
    V = [row[:total_rank] for row in V]
    W = [row[:total_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
