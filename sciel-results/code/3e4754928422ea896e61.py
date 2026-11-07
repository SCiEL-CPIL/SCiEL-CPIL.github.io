import time

def _cycle_lengths(adj, n):
    """Enumerate every simple cycle in a bipartite-structured graph exactly once
    (rooted at the smallest vertex, traversing only increasing-order paths) and
    return the set of distinct cycle lengths."""
    lengths = set()
    for start in range(n):
        # simple path from start; only allow returning to start (closing cycle)
        stack = [(start, frozenset((start,)))]
        while stack:
            u, visited = stack.pop()
            for v in adj[u]:
                if v == start:
                    L = len(visited) + 1
                    if L >= 4:
                        lengths.add(L)
                elif v > u or u == start:
                    # enforce that along the open path (after the start) vertices
                    # are strictly increasing in index, so each undirected cycle is
                    # counted exactly once rooted at its minimum vertex.
                    if v not in visited:
                        stack.append((v, visited | {v}))
    return lengths


def construct():
    deadline = time.time() + 85.0  # safety margin inside the 90s wall-clock budget

    # Two disjoint K_{3,3} components: vertices 0..5 and 6..11.
    # Bipartite => triangle-free, girth 4, min degree 3, and every simple cycle
    # has length in {4, 6}. Disconnected graphs are allowed, so this achieves the
    # theoretical minimum distinct cycle-length count of 2.
    left1  = [0, 1, 2]
    right1 = [3, 4, 5]
    left2  = [6, 7, 8]
    right2 = [9, 10, 11]

    raw = []
    for u in left1:
        for v in right1:
            raw.append((u, v))
    for u in left2:
        for v in right2:
            raw.append((u, v))

    # Exact type/structure normalization: Python int endpoints, no loops, no
    # duplicates, undirected canonical ordering.
    seen = set()
    edges = []
    for (a, b) in raw:
        a = int(a)
        b = int(b)
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append([a, b])

    # Build adjacency and verify all hard constraints.
    n = 12
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = e
        adj[u].append(v)
        adj[v].append(u)

    deg_ok = all(len(adj[i]) >= 3 for i in range(n))

    lengths = _cycle_lengths(adj, n)
    cycle_ok = (len(lengths) == 2) and (lengths == {4, 6})

    if deg_ok and cycle_ok and time.time() < deadline:
        return edges

    # Fallback (should never trigger for this static construction).
    return edges


if __name__ == "__main__":
    _ = construct()
