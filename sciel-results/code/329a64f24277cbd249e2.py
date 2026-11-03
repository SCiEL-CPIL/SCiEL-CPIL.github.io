def search(row_idx, row):
    nodes[0] += 1
    if (nodes[0] & 0x3FFFF) == 0 and time.time() > deadline:
        return False
    if nodes[0] > node_budget:
        return False
    if len(row) == k+1:
        if row_idx > 0 and row[-1] < prev_max[0]:
            return False
        prev_max[0] = max(prev_max[0], row[-1])
        all_rows.append(list(row))
        if row_idx == n_rows - 1:
            return True
        if search(row_idx + 1, [0]):
            return True
        all_rows.pop()
        return False
    # place next mark
    last = row[-1]
    rem = (k+1) - len(row)
    if M - last < rem:
        return False
    # global counting
    need = 0
    for i in range(n_rows):
        if i == row_idx:
            p = len(row); rr = rem
        elif i < row_idx:
            continue  # completed, rem 0
        else:
            p = 1; rr = k
        need += rr * p + rr*(rr-1)//2
    if (M - int(used.sum())) < need:
        return False
    opts = options_for_row(row, last)
    if len(opts) == 0:
        return False
    opts_list = opts.tolist()
    if strategy == 0:
        opts_list.sort()
    elif strategy == 1:
        opts_list.sort(reverse=True)
    else:
        rng.shuffle(opts_list)
    for x in opts_list:
        new_diffs = [x - a for a in row]
        for d in new_diffs:
            used[d] = True
        row.append(x)
        if search(row_idx, row):
            return True
        row.pop()
        for d in new_diffs:
            used[d] = False
    return False
