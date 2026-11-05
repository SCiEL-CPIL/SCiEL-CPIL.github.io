@lru_cache
def gen_blocks(i_list, j_list, k_list):
    i_list = tuple(i_list)
    j_list = tuple(j_list)
    k_list = tuple(k_list)
    i, j, k = len(i_list), len(j_list), len(k_list)
    if min(i, j, k) == 1:
        return [('rank1', i_list, j_list, k_list)]
    if i == 2 and j == 2 and k == 2:
        return [('strassen', i_list, j_list, k_list)]
    best_blocks = None
    best_rank = float('inf')
    # split i
    for split in range(1, i):
        i1 = i_list[:split]
        i2 = i_list[split:]
        b1 = gen_blocks(i1, j_list, k_list)
        b2 = gen_blocks(i2, j_list, k_list)
        r = block_rank(b1) + block_rank(b2)
        if r < best_rank:
            best_rank = r
            best_blocks = b1 + b2
    # split j
    for split in range(1, j):
        j1 = j_list[:split]
        j2 = j_list[split:]
        b1 = gen_blocks(i_list, j1, k_list)
        b2 = gen_blocks(i_list, j2, k_list)
        r = block_rank(b1) + block_rank(b2)
        if r < best_rank:
            best_rank = r
            best_blocks = b1 + b2
    # split k
    for split in range(1, k):
        k1 = k_list[:split]
        k2 = k_list[split:]
        b1 = gen_blocks(i_list, j_list, k1)
        b2 = gen_blocks(i_list, j_list, k2)
        r = block_rank(b1) + block_rank(b2)
        if r < best_rank:
            best_rank = r
            best_blocks = b1 + b2
    return best_blocks
