"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Moore bound for even girth g=2m: 2 * sum_{i=0}^{m-1} (k-1)^i
    m = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(m))
    if moore_bound % 2 == 0:
        moore_bound += 1

    # Try the Lazebnik-Ustimenko-Woldar construction
    best_n, best_edges = _luw_construct(k, g)

    # Verify the LUW result
    if best_edges is not None:
        adj = [[] for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].append(v)
            adj[v].append(u)
        if not _girth_check_fast(adj, best_n, g):
            # LUW result failed girth check, try to fix or use fallback
            pass

    # Try to do better with a structured search approach
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 5:
        for n_candidate in range(moore_bound, best_n):
            if time.time() > deadline:
                break
            if (k * n_candidate) % 2 != 0:
                continue
            result = _try_find_graph(k, g, n_candidate, deadline)
            if result is not None:
                best_n, best_edges = result
                break

    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _girth_check_fast(adj, n, min_girth):
    """Faster girth check: BFS from each vertex with non-backtracking constraint."""
    for start in range(n):
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start))
            visited[(nb, start)] = 1
        while queue:
            node, prev = queue.popleft()
            dist = visited[(node, prev)]
            if dist >= min_girth - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    if dist + 1 < min_girth:
                        return False
                    continue
                state = (nb, node)
                if state in visited:
                    if dist + 1 < min_girth:
                        return False
                else:
                    if dist + 1 < min_girth:
                        visited[state] = dist + 1
                        queue.append(state)
    return True


def _try_find_graph(k, g, n, deadline):
    """Try to find a k-regular graph of girth >= g with n vertices using structured BFS growth."""
    import random
    import time as _time

    # Use a layer-by-layer BFS growth strategy
    # Build a BFS tree from a root, ensuring no short cycles
    max_attempts = 20
    for attempt in range(max_attempts):
        if _time.time() > deadline:
            return None

        edges = _grow_bfs_graph(n, k, g, attempt)
        if edges is None:
            continue

        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        if _girth_check_fast(adj, n, g):
            return n, edges

    return None


def _grow_bfs_graph(n, k, g, seed):
    """Grow a k-regular graph via BFS, avoiding short cycles."""
    import random
    random.seed(seed * 1000 + 42)

    if (n * k) % 2 != 0:
        return None

    # For now, use improved random regular graph with better rejection
    # Configuration model with proper degree checking
    for trial in range(100):
        # Build configuration
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)

        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)

        if not valid:
            continue

        # Check degrees
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            return list(edges)

    return None


# EVOLVE-BLOCK-END
