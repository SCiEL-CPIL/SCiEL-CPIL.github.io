def construct():
    import time
    start_time = time.time()
    DEADLINE = 115.0
    n = 16

    def q3_with_edge(adj_set, u, v, n=16):
        """Check if there's a Q3 subgraph using edge (u,v).
        Fix u->0, v->1 in Q3.
        Q3 edges: 0-1,0-2,0-4,1-3,1-5,2-3,2-6,3-7,4-5,4-6,5-7,6-7
        """
        nbrs_u = [x for x in range(n) if (u, x) in adj_set and x != v]
        nbrs_v = [x for x in range(n) if (v, x) in adj_set and x != u]

        for two in nbrs_u:
            for four in nbrs_u:
                if four == two:
                    continue
                for five in nbrs_v:
                    if five == two or five == four:
                        continue
                    if (four, five) not in adj_set:
                        continue
                    for three in nbrs_v:
                        if three in (u, v, two, four, five):
                            continue
                        if (two, three) not in adj_set:
                            continue
                        for six in range(n):
                            if six in (u, v, two, four, five, three):
                                continue
                            if (two, six) not in adj_set:
                                continue
                            if (four, six) not in adj_set:
                                continue
                            used = {u, v, two, four, five, three, six}
                            for seven in range(n):
                                if seven in used:
                                    continue
                                if (three, seven) not in adj_set:
                                    continue
                                if (five, seven) not in adj_set:
                                    continue
                                if (six, seven) not in adj_set:
                                    continue
                                return True
        return False

    # Build initial graph: K_{3,13} + 13-cycle on B
    adj_set = set()
    for a in range(3):
        for b in range(3, 16):
            adj_set.add((a, b))
            adj_set.add((b, a))
    for i in range(3, 15):
        adj_set.add((i, i + 1))
        adj_set.add((i + 1, i))
    adj_set.add((15, 3))
    adj_set.add((3, 15))

    edges = [[u, v] for (u, v) in adj_set if u < v]

    # Greedy edge addition
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    existing = set()
    for u, v in adj_set:
        existing.add((u, v))
        existing.add((v, u))

    candidates = [(i, j) for (i, j) in all_pairs if (i, j) not in existing]
    candidates.sort(key=lambda p: (0 if p[0] >= 3 and p[1] >= 3 else 1, p[0], p[1]))

    for u, v in candidates:
        if time.time() - start_time > DEADLINE:
            break
        adj_set.add((u, v))
        adj_set.add((v, u))
        if not q3_with_edge(adj_set, u, v, n):
            edges.append([u, v])
        else:
            adj_set.discard((u, v))
            adj_set.discard((v, u))

    return edges
