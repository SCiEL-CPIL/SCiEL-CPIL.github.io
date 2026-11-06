    # Start 1: Turan Graph T(20,4)
    adj_turan = [0] * n
    for u in range(n):
        part_u = u // 5
        for v in range(u+1, n):
            part_v = v // 5
            if part_u != part_v:
                adj_turan[u] |= (1 << v)
                adj_turan[v] |= (1 << u)
