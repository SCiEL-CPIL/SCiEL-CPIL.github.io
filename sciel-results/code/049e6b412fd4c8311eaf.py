for c_row in range(2):
    for a_col in range(2):
        w_idx = (b_row * 2 + c_row) * 4 + (ar * 2 + a_col)
        W[w_idx, col] += 1
