    for d in range(1, N + 1):
        for a in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                # Add to conflicts for a, b, c, e
                conflicts[a].append((b, c, e))
                conflicts[b].append((a, c, e))
                conflicts[c].append((a, b, e))
                conflicts[e].append((a, b, c))
