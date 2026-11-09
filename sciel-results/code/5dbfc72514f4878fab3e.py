def eval_p(p, adj_list, n):
    order = np.argsort(-p)
    selected = [False] * n
    count = 0
    for v in order:
        if not selected[v]:
            count += 1
            selected[v] = True
            for u in adj_list[v]:
                selected[u] = True
    return count
