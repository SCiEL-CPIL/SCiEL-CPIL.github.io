    perm_U = [0] * 16
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    r1 = i1 * 2 + j1
                    r2 = i2 * 2 + j2
                    kron_row = r1 * 4 + r2
                    global_row = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    perm_U[kron_row] = global_row
