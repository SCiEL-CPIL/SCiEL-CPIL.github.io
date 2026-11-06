def get_idx(i, j):
    if i > j: i, j = j, i
    return i * (n - 1) - i * (i - 1) // 2 + j - i - 1
