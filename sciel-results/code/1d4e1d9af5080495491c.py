    # Precompute squared distances
    dist2 = [[0]*n for _ in range(n)]
    for i in range(n):
        pi = all_points[i]
        for j in range(n):
            pj = all_points[j]
            dist2[i][j] = _dist2(pi, pj)

    for i in range(n):
        for j in range(i + 1, n):
            is_conflict = False
            # Check all k
            # We can break early
            for k in range(n):
                if k == i or k == j:
                    continue
                if dist2[i][j] == dist2[i][k] or dist2[i][j] == dist2[j][k] or dist2[i][k] == dist2[j][k]:
                    is_conflict = True
                    break
            if is_conflict:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
