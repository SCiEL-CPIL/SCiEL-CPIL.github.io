import random
import time

def construct():
    n = 15
    deadline = time.time() + 80  # safety margin before 90s kill
    best_edges = None
    best_count = -1
    best_degrees = None

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(all_edges)

    def check_valid(adj, deg):
        """Check if graph is valid: no path s-u-v-t with s!=t has deg(s)==deg(t)."""
        for (u, v) in adj:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def get_violations(adj, deg):
        """Return set of edges that participate in at least one violation."""
        viol_edges = set()
        for (u, v) in adj:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        viol_edges.add((u, v))
                        break
                if (u, v) in viol_edges:
                    break
        return viol_edges

    def try_remove_edge(adj, deg, edge):
        """Try removing an edge and return (new_adj, new_deg, valid) or None."""
        u, v = edge
        new_adj = [list(x) for x in adj]
        new_adj[u].remove(v)
        new_adj[v].remove(u)
        new_deg = list(deg)
        new_deg[u] -= 1
        new_deg[v] -= 1
        if check_valid(new_adj, new_deg):
            return (new_adj, new_deg)
        return None

    # Strategy 1: Start dense, remove violations greedily
    for attempt in range(200):
        if time.time() > deadline:
            break

        random.seed(attempt * 12345 + 67890)
        # Start with a dense random graph (keep ~70% of edges)
        adj = [[] for _ in range(n)]
        kept_edges = []
        for (i, j) in all_edges:
            if random.random() < 0.70:
                adj[i].append(j)
                adj[j].append(i)
                kept_edges.append((i, j))

        deg = [len(adj[i]) for i in range(n)]

        # Iteratively remove violating edges
        max_iters = 500
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not kept_edges:
                break
            if check_valid(adj, deg):
                break
            viol_edges = get_violations(adj, deg)
            if not viol_edges:
                break

            # Find the edge whose removal is "cheapest" (removes fewest other edges' potential)
            # Heuristic: remove edge with minimum (deg[u] + deg[v]) to preserve structure
            candidates = [e for e in viol_edges if e in set(kept_edges)]
            if not candidates:
                # Try removing any violating edge
                candidates = list(viol_edges)
            if not candidates:
                break

            # Pick the edge with minimum sum of endpoint degrees
            best_e = min(candidates, key=lambda e: deg[e[0]] + deg[e[1]])
            result = try_remove_edge(adj, deg, best_e)
            if result is None:
                # Try next candidate
                tried = {best_e}
                found = False
                for e in sorted(candidates, key=lambda x: deg[x[0]] + deg[x[1]]):
                    if e in tried:
                        continue
                    result = try_remove_edge(adj, deg, e)
                    if result is not None:
                        best_e = e
                        found = True
                        break
                if not found:
                    # Try all remaining edges in the graph
                    for e in list(kept_edges):
                        if e in tried:
                            continue
                        result = try_remove_edge(adj, deg, e)
                        if result is not None:
                            best_e = e
                            found = True
                            break
                    if not found:
                        break

            adj, deg = result
            kept_edges.remove(best_e)

        if check_valid(adj, deg):
            count = len(kept_edges)
            if count > best_count:
                best_count = count
                best_edges = [list(e) for e in kept_edges]
                best_degrees = list(deg)

        if best_count >= 55:  # good enough
            break

    # Strategy 2: Structured construction based on degree classes
    # Design: 3 high-degree vertices (deg 10-12), 6 medium (deg 4-6), 6 low (deg 1-3)
    # High vertices connect to everything; medium connect to high + some low; low connect to high + few medium
    if time.time() < deadline:
        # Try multiple structured configurations
        configs = [
            # (high_count, med_count, low_count, high_conn_frac, med_low_frac)
            (3, 6, 6, 1.0, 0.3),
            (3, 5, 7, 1.0, 0.25),
            (4, 5, 6, 1.0, 0.3),
            (2, 7, 6, 1.0, 0.35),
            (3, 6, 6, 0.95, 0.25),
        ]

        for cfg_idx, (h, m, l, hf, ml) in enumerate(configs):
            if time.time() > deadline:
                break
            if h + m + l != n:
                continue

            high = list(range(h))
            med = list(range(h, h + m))
            low = list(range(h + m, n))

            adj = [[] for _ in range(n)]
            edges = []

            # High-high edges
            for i in range(h):
                for j in range(i + 1, h):
                    adj[i].append(j)
                    adj[j].append(i)
                    edges.append((i, j))

            # High-med edges
            for hi in high:
                for mi in med:
                    if random.random() < hf:
                        adj[hi].append(mi)
                        adj[mi].append(hi)
                        edges.append((hi, mi))

            # High-low edges
            for hi in high:
                for li in low:
                    if random.random() < hf:
                        adj[hi].append(li)
                        adj[li].append(hi)
                        edges.append((hi, li))

            # Med-med edges (few)
            for i in range(m):
                for j in range(i + 1, m):
                    if random.random() < 0.15:
                        a = med[i]
                        b = med[j]
                        adj[a].append(b)
                        adj[b].append(a)
                        edges.append((a, b))

            # Med-low edges
            for mi in med:
                for li in low:
                    if random.random() < ml:
                        adj[mi].append(li)
                        adj[li].append(mi)
                        edges.append((mi, li))

            # Low-low edges (very few)
            for i in range(l):
                for j in range(i + 1, l):
                    if random.random() < 0.05:
                        a = low[i]
                        b = low[j]
                        adj[a].append(b)
                        adj[b].append(a)
                        edges.append((a, b))

            deg = [len(adj[i]) for i in range(n)]

            # Fix violations
            for _ in range(300):
                if time.time() > deadline:
                    break
                if not edges:
                    break
                if check_valid(adj, deg):
                    break
                viol_edges = get_violations(adj, deg)
                if not viol_edges:
                    break
                candidates = [e for e in viol_edges if e in set(edges)]
                if not candidates:
                    break
                best_e = min(candidates, key=lambda e: deg[e[0]] + deg[e[1]])
                result = try_remove_edge(adj, deg, best_e)
                if result is None:
                    found = False
                    for e in candidates:
                        if e == best_e:
                            continue
                        result = try_remove_edge(adj, deg, e)
                        if result is not None:
                            best_e = e
                            found = True
                            break
                    if not found:
                        for e in list(edges):
                            result = try_remove_edge(adj, deg, e)
                            if result is not None:
                                best_e = e
                                found = True
                                break
                        if not found:
                            break
                adj, deg = result
                edges.remove(best_e)

            if check_valid(adj, deg):
                count = len(edges)
                if count > best_count:
                    best_count = count
                    best_edges = [list(e) for e in edges]
                    best_degrees = list(deg)

    # Strategy 3: Start from complement (sparse), add edges
    if time.time() < deadline:
        for attempt in range(100):
            if time.time() > deadline:
                break
            random.seed(attempt * 9999 + 1234)
            adj = [[] for _ in range(n)]
            edges = []

            # Start with a random sparse graph
            for (i, j) in all_edges:
                if random.random() < 0.3:
                    adj[i].append(j)
                    adj[j].append(i)
                    edges.append((i, j))

            deg = [len(adj[i]) for i in range(n)]

            # Try to add edges greedily
            remaining = [(i, j) for (i, j) in all_edges if (i, j) not in set(edges)]
            random.shuffle(remaining)

            for (i, j) in remaining:
                if time.time() > deadline:
                    break
                # Try adding this edge
                new_adj = [list(x) for x in adj]
                new_adj[i].append(j)
                new_adj[j].append(i)
                new_deg = list(deg)
                new_deg[i] += 1
                new_deg[j] += 1
                if check_valid(new_adj, new_deg):
                    adj = new_adj
                    deg = new_deg
                    edges.append((i, j))

            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = [list(e) for e in edges]
                best_degrees = list(deg)

            if best_count >= 55:
                break

    if best_edges is None:
        # Fallback: star graph (always valid)
        best_edges = [[0, v] for v in range(1, n)]
    else:
        # Ensure types are correct
        best_edges = [[int(u), int(v)] for u, v in best_edges]

    return best_edges
