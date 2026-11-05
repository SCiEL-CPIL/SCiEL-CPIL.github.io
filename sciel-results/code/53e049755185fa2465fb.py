def construct():
    import time
    import itertools

    n = 16
    start = time.time()
    deadline = 110.0

    def has_q3_full(adj):
        for verts in itertools.combinations(range(n), 8):
            deg = [0] * 8
            edges = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    if adj[verts[i]][verts[j]]:
                        edges += 1
                        deg[i] += 1
                        deg[j] += 1
            if edges != 12 or any(d != 3 for d in deg):
                continue
            color = [-1] * 8
            color[0] = 0
            stack = [0]
            ok = True
            while stack:
                u = stack.pop()
                for v in range(8):
                    if adj[verts[u]][verts[v]]:
                        if color[v] < 0:
                            color[v] = 1 - color[u]
                            stack.append(v)
                        elif color[v] == color[u]:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                return True
        return False

    def build_adj(per_edges):
        adj = [[False] * n for _ in range(n)]
        for a in range(3):
            for b in range(3, n):
                adj[a][b] = True
                adj[b][a] = True
        for u, v in per_edges:
            uu, vv = 3 + u, 3 + v
            adj[uu][vv] = True
            adj[vv][uu] = True
        return adj

    cycle = set([(i, (i + 1) % 13) for i in range(13)])
    missing = [(i, j) for i in range(13) for j in range(i + 1, 13) if (i, j) not in cycle]

    for u, v in missing:
        if time.time() > deadline:
            break
        trial = cycle | {(u, v)}
        adj = build_adj(trial)
        if not has_q3_full(adj):
            cycle = trial

    edges = []
    for a in range(3):
        for b in range(3, n):
            edges.append([a, b])
    for u, v in sorted(cycle):
        edges.append([3 + u, 3 + v])

    return edges
