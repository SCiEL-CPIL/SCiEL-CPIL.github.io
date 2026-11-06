"""Decompose fixed instance graphs into few cycles + edges (Erdős-Gallai style)."""
import time
from collections import defaultdict, deque
from random import Random

# EVOLVE-BLOCK-START
def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n, time_limit=10.0):
    """
    Greedily extract edge-disjoint cycles, preferring longer cycles (more edge savings).
    Uses BFS to find longest path between endpoints of each remaining edge.
    """
    if not edges:
        return []

    adj = defaultdict(set)
    alive = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        alive.add((u, v))
        alive.add((v, u))

    pieces = []
    start_time = time.time()

    def bfs_longest_path(s, t, banned_edge):
        """
        BFS from s to t, avoiding using banned_edge.
        Returns the path (list of vertices) if found, else None.
        We use BFS which gives shortest path, but we want longest.
        For small graphs, we can try to find a longer path by using DFS or just BFS.
        Actually, for cycle decomposition, finding ANY cycle is good, but longer cycles
        save more pieces. Let's use a simple approach: BFS gives shortest path, which
        gives the shortest cycle. To get longer cycles, we could use DFS, but that's
        expensive. Given the time limit, let's stick with BFS for speed and reliability.
        """
        prev = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for v in sorted(adj[u]):
                if v in prev:
                    continue
                # Check if edge (u,v) is the banned edge
                if (u, v) == banned_edge or (v, u) == banned_edge:
                    continue
                prev[v] = u
                q.append(v)
        if t not in prev:
            return None
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    # Greedily find cycles
    # Strategy: repeatedly find the edge (a,b) that yields the longest cycle (longest path from a to b not using (a,b))
    # To keep it efficient, we can just iterate through alive edges and pick the best one each time.
    # For n<=30, this is fast enough.

    while True:
        if time.time() - start_time > time_limit:
            break
        if not alive:
            break

        best_edge = None
        best_path = None
        best_len = -1

        # Iterate over unique edges
        checked = set()
        for a, b in list(alive):
            if a > b:
                continue
            if (a, b) in checked:
                continue
            checked.add((a, b))

            # Check if edge is still alive
            if (a, b) not in alive:
                continue

            path = bfs_longest_path(a, b, (a, b))
            if path is None:
                continue
            L = len(path) - 1  # number of edges in path
            if L < 2:  # Need at least 3 vertices in cycle, so path must have at least 2 edges
                continue
            if L > best_len:
                best_len = L
                best_edge = (a, b)
                best_path = path

        if best_edge is None:
            break

        a, b = best_edge
        # Remove edges from best_path and the closing edge (a,b)
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            alive.discard((u, v))
            alive.discard((v, u))
            adj[u].discard(v)
            adj[v].discard(u)

        # Remove closing edge
        alive.discard((a, b))
        alive.discard((b, a))
        adj[a].discard(b)
        adj[b].discard(a)

        # Add cycle piece
        pieces.append([int(v) for v in best_path])

    # Add remaining edges as single-edge pieces
    remaining_edges = set()
    for u, v in alive:
        if u < v:
            remaining_edges.add((u, v))

    for u, v in sorted(remaining_edges):
        pieces.append([u, v])

    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    start_time = time.time()
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        # Allocate time for this graph, leaving buffer for remaining graphs and safety margin
        # Total limit 150s, use 145s for computation
        remaining_time = 145.0 - elapsed
        graphs_left = len(specs) - i
        alloc_time = remaining_time / graphs_left
        # Cap per-graph time to avoid one graph taking too long, but ensure at least 1s
        alloc_time = max(1.0, min(alloc_time, 20.0))

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, time_limit=alloc_time)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
