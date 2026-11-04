    def place_strassen(u_idx, v_idx, w_idx, base_col):
        tl, tr, bl, br = u_idx
        tl_v, tr_v, bl_v, br_v = v_idx
        tl_w, tr_w, bl_w, br_w = w_idx

        # P1 = (A00+A11)(B00+B11)
        # C00 += P1, C11 += P1
        U_mat[tl, base_col + 0] += 1.0
        U_mat[br, base_col + 0] += 1.0
        V_mat[tl_v, base_col + 0] += 1.0
        V_mat[br_v, base_col + 0] += 1.0
        W_mat[tl_w, base_col + 0] += 1.0
        W_mat[br_w, base_col + 0] += 1.0

        # P2 = (A10+A11)B00
        # C11 -= P2
        U_mat[bl, base_col + 1] += 1.0
        U_mat[br, base_col + 1] += 1.0
        V_mat[tl_v, base_col + 1] += 1.0
        W_mat[br_w, base_col + 1] -= 1.0

        # P3 = A00(B01-B11)
        # C01 += P3, C00 -= P3
        U_mat[tl, base_col + 2] += 1.0
        V_mat[tr_v, base_col + 2] += 1.0
        V_mat[br_v, base_col + 2] -= 1.0
        W_mat[tr_w, base_col + 2] += 1.0
        W_mat[tl_w, base_col + 2] -= 1.0

        # P4 = A11(B10-B00)
        # C10 += P4, C00 -= P4
        U_mat[br, base_col + 3] += 1.0
        V_mat[bl_v, base_col + 3] += 1.0
        V_mat[tl_v, base_col + 3] -= 1.0
        W_mat[bl_w, base_col + 3] += 1.0
        W_mat[tl_w, base_col + 3] -= 1.0

        # P5 = (A00+A01)B11
        # C10 += P5, C11 += P5
        U_mat[tl, base_col + 4] += 1.0
        U_mat[tr, base_col + 4] += 1.0
        V_mat[br_v, base_col + 4] += 1.0
        W_mat[bl_w, base_col + 4] += 1.0
        W_mat[br_w, base_col + 4] += 1.0

        # P6 = (A10-A00)(B00+B01)
        # C00 += P6
        U_mat[bl, base_col + 5] += 1.0
        U_mat[tl, base_col + 5] -= 1.0
        V_mat[tl_v, base_col + 5] += 1.0
        V_mat[tr_v, base_col + 5] += 1.0
        W_mat[tl_w, base_col + 5] += 1.0

        # P7 = (A01-A11)(B10+B11)
        # C11 -= P7
        U_mat[tr, base_col + 6] += 1.0
        U_mat[br, base_col + 6] -= 1.0
        V_mat[bl_v, base_col + 6] += 1.0
        V_mat[br_v, base_col + 6] += 1.0
        W_mat[br_w, base_col + 6] -= 1.0
