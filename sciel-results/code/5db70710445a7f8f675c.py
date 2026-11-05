def get_idx(i, j):
    if i > j: i, j = j, i
    return idx_map[(i, j)]
