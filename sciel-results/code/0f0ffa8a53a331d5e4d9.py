            row_i = colors[i]
            row_j = colors[j]
            for k in range(n_target):
                if k == i or k == j: continue
                c_ik = row_i[k]
                c_jk = row_j[k]
                # ...
