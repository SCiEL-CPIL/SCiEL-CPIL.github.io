import time

N = 15

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    # Deterministic seed: 5 disjoint triangles (guaranteed valid, 15 edges)
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N

    def add(u, v):
        adj[u][v] = adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1

    def rem(u, v):
        adj[u][v] = adj[v][u] = 0
        deg[u] -= 1
        deg[v] -= 1

    # triangles: (0,1,2),(3,4,5),(6,7,8),(9,10,11),(12,13,14)
    for a in range(0, N, 3):
        add(a, a+1); add(a+1, a+2); add(a, a+2)

    def valid():
        # For every edge (u,v) and s in N(u)\{v}, t in N(v)\{u}, s!=t:
        # require deg(s) != deg(t)
        for u in range(N):
            for v in range(u+1, N):
                if not adj[u][v]:
                    continue
                nu = [s for s in range(N) if adj[u][s] and s != v]
                nv = [t for t in range(N) if adj[v][t] and t != u]
                for s in nu:
                    ds = deg[s]
                    for t in nv:
                        if t == s:
                            continue
                        if ds == deg[t]:
                            return False
        return True

    # Local improvement: add safe edges, remove unsafe edges (if any), repeat
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # pass 1: add safe missing edges
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                add(u, v)
                if valid():
                    improved = True
                else:
                    rem(u, v)
        # pass 2: remove any existing edge that violates (shouldn't happen, but be safe)
        for u in range(N):
            for v in range(u+1, N):
                if not adj[u][v]:
                    continue
                # check this edge alone against current graph
                nu = [s for s in range(N) if adj[u][s] and s != v]
                nv = [t for t in range(N) if adj[v][t] and t != u]
                bad = False
                for s in nu:
                    ds = deg[s]
                    for t in nv:
                        if t == s and False:
                            continue
                        if t == s:
                            continue
                        if ds == deg[t]:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    rem(u, v)
                    improved = True
        if time.time() > deadline:
            break

    # Final safety: if invalid, drop edges until valid (deterministic)
    guard = 0
    while not valid() and guard < 5000:
        guard += 1
        # find a violating edge and remove it
        for u in range(N):
            for v in range(u+1, N):
                if not adj[u][v]:
                    continue
                nu = [s for s in range(N) if adj[u][s] and s != v]
                nv = [t for t in range(N) if adj[v][t] and t != u]
                for s in nu:
                    ds = deg[s]
                    for t in nv:
                        if t == s:
                            continue
                        if ds == deg[t]:
                            rem(u, v)
                            break
                    if not adj[u][v]:
                        break
                if not adj[u][v]:
                    break
            if not adj[u][v]:
                break
        if time.time() > deadline:
            break

    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                edges.append([int(i), int(j)])
    return edges
