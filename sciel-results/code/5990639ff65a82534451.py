import time, random

def _is_valid(adj, deg, n):
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for s in adj[u]:
                    if s == v:
                        continue
                    ds = deg[s]
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if ds == deg[t]:
                            return False
    return True

def construct():
    n = 15
    deadline = time.time() + 80.0
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best_edges = None
    best_count = -1

    rng = random.Random(12345)
    # multiple restarts with different orders
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        order = all_pairs[:]
        rng.shuffle(order)
        adj = [set() for _ in range(n)]
        for v in range(1, n):
            adj[0].add(v); adj[v].add(0)
        deg = [len(adj[i]) for i in range(n)]
        improved = True
        while improved:
            improved = False
            for (u, v) in order:
                if time.time() > deadline - 5.0:
                    break
                if v in adj[u]:
                    continue
                adj[u].add(v); adj[v].add(u)
                deg[u] += 1; deg[v] += 1
                if _is_valid(adj, deg, n):
                    improved = True
                else:
                    adj[u].discard(v); adj[v].discard(u)
                    deg[u] -= 1; deg[v] -= 1
            if time.time() > deadline - 5.0:
                break
        cnt = sum(len(a) for a in adj) // 2
        if cnt > best_count:
            best_count = cnt
            best_edges = [[int(u), int(v)] for u in range(n) for v in adj[u] if u < v]

    if best_edges is None:
        best_edges = [[0, v] for v in range(1, n)]
    return best_edges
