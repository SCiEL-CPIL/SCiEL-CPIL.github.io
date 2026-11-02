    col_offset = 0
    for t, a_coeffs, b_coeffs in strassen_coeffs:
        for r in range(m3_rank):
            col = col_offset + r
            # ...
            col_offset += 1
