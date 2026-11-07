import time

def construct():
    start_time = time.time()
    deadline = 58.0

    # Two disjoint K3,3 components
    # Vertices 0-5: A={0,1,2}, B={3,4,5}
    # Vertices 6-11: A={6,7,8}, B={9,10,11}
    edges = []
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([int(u), int(v)])
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([int(u), int(v)])

    # Quick validity checks
    n = 12
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    if min(deg) < 3:
        return []

    # Triangle-free check
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    for i in range(n):
        for j in adj[i]:
            if i < j and (adj[i] & adj[j]):
                return []

    # Confirm at least one cycle exists (K3,3 always has 4-cycles)
    # Simple check: any edge (u,v) where u and v share a neighbor gives a 4-cycle
    # K3,3 guarantees this.

    if time.time() - start_time > deadline:
        return edges

    return edges
