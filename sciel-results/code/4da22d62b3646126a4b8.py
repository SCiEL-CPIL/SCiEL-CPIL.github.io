        for i in range(6):
            i_a, i_b = divmod(i, 3)
            for j in range(6):
                j_a, j_b = divmod(j, 3)
                row_u = i * 6 + j
                for r_a in range(7):
                    for r_b in range(27):
                        col = r_a * 27 + r_b
                        u2_val = U2[i_a * 2 + j_a, r_a]
                        u3_val = U3[i_b * 3 + j_b, r_b]
                        U6[row_u, col] = u2_val * u3_val
        