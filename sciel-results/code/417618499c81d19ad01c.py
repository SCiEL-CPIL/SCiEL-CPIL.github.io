import time

def construct():
    n = 15
    deadline = time.time() + 55

    # Start with K_{7,8}: vertices 0..6 in part A, 7..14 in part B
    part_a = list(range(7))
    part_b = list(range(7, 15))

    adj = [set() for _ in range(n)]
    deg = [0] * n

    for u in part_a:
        for v in part_b:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    missing_edges = [(u, v) for (u, v) in all_edges if v not in adj[u]]

    # Deterministic order: try edges that connect vertices with more "room"
    # First, try intra-part edges, then cross-part missing (there are none for K_{7,8})
    # Sort by a heuristic: prefer edges where both endpoints have lower current degree
    # (lower degree means fewer potential conflicts)
    missing_edges.sort(key=lambda e: (deg[e[0]] + deg[e[1]], e[0], e[1]))

    def check_new_violations(u, v, adj, deg):
        """Check if adding edge (u,v) creates any 3-edge path with equal-degree endpoints."""
        # Path s-u-v-t: (u,v) is middle edge. Violation if deg(s)==deg(t)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        # Path u-v-y-t: (u,v) is first edge. Violation if deg(u)==deg(t)
        for y in adj[v]:
            if y == u:
                continue
            for t in adj[y]:
                if t == u or t == v:
                    continue
                if deg[u] == deg[t]:
                    return False
        # Path s-x-u-v: (u,v) is last edge. Violation if deg(s)==deg(v)
        for x in adj[u]:
            if x == v:
                continue
            for s in adj[x]:
                if s == u or s == v:
                    continue
                if deg[s] == deg[v]:
                    return False
        return True

    added = 0
    for (u, v) in missing_edges:
        if time.time() > deadline:
            break
        if check_new_violations(u, v, adj, deg):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            added += 1
            # After adding an edge, re-sort remaining candidates? No, too expensive.
            # Instead, just continue with the pre-sorted list.

    # Second pass: try edges again in reverse order (might unlock new possibilities)
    missing_edges2 = [(u, v) for (u, v) in all_edges if v not in adj[u]]
    missing_edges2.sort(key=lambda e: (deg[e[0]] + deg[e[1]], e[0], e[1]), reverse=True)
    for (u, v) in missing_edges2:
        if time.time() > deadline:
            break
        if check_new_violations(u, v, adj, deg):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            added += 1

    # Third pass: another forward pass
    missing_edges3 = [(u, v) for (u, v) in all_edges if v not in adj[u]]
    missing_edges3.sort(key=lambda e: (deg[e[0]] + deg[e[1]], e[0], e[1]))
    for (u, v) in missing_edges3:
        if time.time() > deadline:
            break
        if check_new_violations(u, v, adj, deg):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            added += 1

    # Fourth pass: try all remaining edges in numerical order
    missing_edges4 = [(u, v) for (u, v) in all_edges if v not in adj[u]]
    for (u, v) in missing_edges4:
        if time.time() > deadline:
            break
        if check_new_violations(u, v, adj, deg):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            added += 1

    result = []
    for u in range(n):
        for v in sorted(adj[u]):
            if u < v:
                result.append([int(u), int(v)])

    return result
