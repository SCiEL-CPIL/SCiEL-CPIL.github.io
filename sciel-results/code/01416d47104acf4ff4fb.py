def construct():
    import time
    import random

    start_time = time.time()
    DEADLINE = 110.0

    n = 40
    random.seed(12345)

    best_edges = None
    best_count = 0

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def edges_from_adj(adj):
        return [[u, v] for u in range(n) for v in sorted(adj[u]) if u < v]

    def has_c6_through(adj, u, v):
        """Check if adding edge (u,v) to adj creates a 6-cycle."""
        # Need a path v-a-b-c-d-u of length 4 in adj (5 edges total with (u,v))
        for a in adj[v]:
            if a == u:
                continue
            for b in adj[a]:
                if b == u or b == v:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == a:
                        continue
                    for d in adj[c]:
                        if d == u or d == v or d == a or d == b:
                            continue
                        if u in adj[d]:
                            return True
        return False

    def add_all_possible(adj, deadline):
        """Iteratively add all edges that don't create a C6."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(n):
                if time.time() > deadline:
                    return
                for j in range(i + 1, n):
                    if j not in adj[i]:
                        if not has_c6_through(adj, i, j):
                            adj[i].add(j)
                            adj[j].add(i)
                            changed = True

    def greedy_random(adj, deadline):
        """Add edges in random order."""
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if j not in adj[i]]
        random.shuffle(candidates)
        for u, v in candidates:
            if time.time() > deadline:
                break
            if not has_c6_through(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)

    # Strategy 1: Start from 8 disjoint K5 (80 edges)
    adj = [set() for _ in range(n)]
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                adj[5 * k + a].add(5 * k + b)
                adj[5 * k + b].add(5 * k + a)

    add_all_possible(adj, min(DEADLINE - 5, start_time + 15.0))
    count = count_edges(adj)
    if count > best_count:
        best_count = count
        best_edges = edges_from_adj(adj)

    # Strategy 2: Multiple random restarts from scratch
    for r in range(60):
        if time.time() > DEADLINE - 10:
            break

        adj = [set() for _ in range(n)]
        greedy_random(adj, min(DEADLINE - 10, start_time + 1.5))
        add_all_possible(adj, min(DEADLINE - 5, start_time + 3.0))

        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_edges = edges_from_adj(adj)

    if best_edges is None:
        best_edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    best_edges.append([5 * k + a, 5 * k + b])

    return best_edges
