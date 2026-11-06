        if abs(dp_table[i][j] - val_taken) < 1e-9:
             taken_indices.append(i-1)
             j -= wi
             continue
