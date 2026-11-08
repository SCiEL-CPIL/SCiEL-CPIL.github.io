    def add_rank1(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        if col >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1
