   def add_strassen_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset):
        # M1: (A11+A22)(B11+B22) -> C11, C22
        u = {u_idx(a_i_offset+0, a_j_offset+0): 1, u_idx(a_i_offset+1, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+0, b_k_offset+0): 1, v_idx(b_j_offset+1, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): 1, w_idx(c_k_offset+1, c_i_offset+1): 1}
   