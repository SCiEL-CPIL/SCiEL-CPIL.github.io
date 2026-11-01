"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import random as _random
from collections import deque as _deque

# EVOLVE-BLOCK-START
import sys as _sys

def construct(k, g):
    """Build a k-regular graph with girth at least g, minimizing vertex count.
    Returns (n, edges)."""
    deadline = _time.monotonic() + 125  # safety margin under 150s

    # ---- Step 1: Try explicit constructions ----
    result = _try_luw(k, g)
    if result is not None:
        n_luw, edges_luw = result
    else:
        n_luw, edges_luw = _sys.maxsize, []

    # ---- Step 2: Try girth-constrained BFS construction with repair ----
    best_n = n_luw
    best_edges = edges_luw

    if k >= 3 and g >= 4 and _time.monotonic() < deadline - 5:
        # Moore bound: minimum possible n
        if g % 2 == 1:  # odd girth: (k*(k^(d)-1)/(k-1)+1) where g=2d+1
            d = (g - 1) // 2
            moore = 1 + k * sum(k**i for i in range(d))
        else:  # even girth: g=2d
            d = g // 2
            moore = 2 * sum(k**i for i in range(d))

        # For small instances, try randomized construction with backtracking
        if moore <= 200 and _time.monotonic() < deadline - 10:
            t0 = _time.monotonic()
            result2 = _construct_bfs_repair(k, g, deadline - 5, seed=42)
            if result2 is not None:
                n2, edges2 = result2
                if n2 < best_n:
                    best_n, best_edges = n2, edges2

        # Try a few more random seeds if time permits
        attempt = 0
        while _time.monotonic() < deadline - 15 and attempt < 50:
            attempt += 1
            result3 = _construct_bfs_repair(k, g, deadline - 10, seed=attempt * 137 + 7)
            if result3 is not None:
                n3, edges3 = result3
                if n3 < best_n:
                    best_n, best_edges = n3, edges3
            if _time.monotonic() < deadline - 20:
                result4 = _construct_luw_variant(k, g, attempt)
                if result4 is not None:
                    n4, edges4 = result4
                    if n4 < best_n:
                        best_n, best_edges = n4, edges4

    # ---- Step 3: Return the best result ----
    if best_edges:
        return best_n, best_edges
    # Fallback: LUW result
    if n_luw < _sys.maxsize:
        return n_luw, edges_luw
    # Last resort: cycle C_k if k>=3 (girth = k)
    if k >= 3 and g <= k:
        n = k
        edges = [(i, (i+1) % k) for i in range(k)]
        return n, edges
    # Complete graph K_k (girth 3)
    n = k
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    return n, edges


def _try_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    if q < 3:
        return None
    # Need d such that girth >= d + 5 (odd) or similar
    # For odd d: girth = d + 5 (if d odd) or d + 6
    # We want girth >= g, so need d + 5 >= g => d >= g - 5
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # use odd d for guaranteed girth d+5

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
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

    # Verify girth
    if _verify_girth(len(order), edges, g):
        return len(order), edges
    return None


def _construct_luw_variant(k, g, variant):
    """Try LUW with different parameters."""
    q = k
    if q < 3:
        return None
    for d_offset in [0, 1, 2]:
        d = g - 5 + d_offset
        if d < 3:
            d = 3
        if d % 2 == 0:
            d += 1

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q
            sub_fn = lambda x, y: (x - y) % q
            mul_fn = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5 + variant) % 4
            c0 = c - (c - 5 + variant) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
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

        if _verify_girth(len(order), edges, g):
            return len(order), edges
    return None


def _construct_bfs_repair(k, g, deadline, seed=0):
    """Girth-constrained BFS layer construction with backtracking repair.
    Build vertex by vertex, ensuring no short cycles are created."""
    rng = _random.Random(seed)

    # Target: start from Moore bound, grow if needed
    if g % 2 == 1:
        d = (g - 1) // 2
        target_n = 1 + k * sum(k**i for i in range(d))
    else:
        d = g // 2
        target_n = 2 * sum(k**i for i in range(d))

    # Build adjacency list
    adj = {}
    degree = {}
    edge_set = set()

    n = target_n
    for i in range(n):
        adj[i] = []
        degree[i] = 0

    # Phase 1: Greedy assignment - give each vertex k neighbors avoiding short cycles
    # Use a queue-based approach: process vertices, assign neighbors
    order = list(range(n))
    rng.shuffle(order)

    for v in order:
        if degree[v] >= k:
            continue
        needed = k - degree[v]
        candidates = [u for u in range(n) if u != v and u not in adj[v] and (v, u) not in edge_set]
        rng.shuffle(candidates)

        for u in candidates:
            if degree[v] >= k:
                break
            if _would_create_short_cycle(v, u, adj, g):
                continue
            adj[v].append(u)
            adj[u].append(v)
            degree[v] += 1
            degree[u] += 1
            edge_set.add((min(v, u), max(v, u)))

    # Check if we succeeded
    if all(degree[v] == k for v in range(n)):
        edges = list(edge_set)
        if _verify_girth(n, edges, g):
            return n, edges
        # Girth verification failed, try repair
        if _time.monotonic() < deadline:
            repaired = _repair_graph(n, adj, k, g, deadline, rng)
            if repaired is not None:
                return n, repaired
        return None

    # Phase 2: If greedy failed, try adding vertices
    # This is a simplified fallback - just return None to let LUW handle it
    return None


def _would_create_short_cycle(v, u, adj, g):
    """Check if adding edge (v,u) would create a cycle shorter than g.
    Uses BFS from v (excluding u) to see if u is reachable in < g-1 steps."""
    if g <= 3:
        return False  # Can't avoid triangles easily

    # BFS from v, not going through u, see if we can reach u in < g-1 steps
    visited = {v: 0}
    queue = _deque([(v, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= g - 1:
            break
        for nb in adj[node]:
            if nb == u:
                # Found a path from v to u of length dist+1
                # Adding edge (v,u) creates cycle of length dist+2
                if dist + 2 < g:
                    return True
                continue
            if nb not in visited:
                visited[nb] = dist + 1
                queue.append((nb, dist + 1))
    return False


def _repair_graph(n, adj, k, g, deadline, rng):
    """Try to fix girth violations by swapping edges."""
    # Find short cycles and try to break them
    for iteration in range(100):
        if _time.monotonic() > deadline:
            break

        # Find a short cycle
        cycle = _find_short_cycle(n, adj, g)
        if cycle is None:
            edges = []
            for v in range(n):
                for u in adj[v]:
                    if v < u:
                        edges.append((v, u))
            if _verify_girth(n, edges, g):
                return edges
            return None

        # Try to break the cycle by swapping an edge
        # Remove one edge from the cycle, reconnect
        broken = False
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i+1) % len(cycle)]
            # Try to find alternative neighbors
            for a in adj[a][:]:
                if a == b:
                    continue
                # Find a new neighbor for a
                candidates = [x for x in range(n) if x not in adj[a] and x != b]
                rng.shuffle(candidates)
                for new_nb in candidates:
                    if new_nb in adj[b]:
                        continue
                    if _would_create_short_cycle(a, new_nb, adj, g):
                        continue
                    # Do the swap
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[a].append(new_nb)
                    adj[new_nb].append(a)
                    broken = True
                    break
                if broken:
                    break
            if broken:
                break

        if not broken:
            break

    # Final check
    edges = []
    for v in range(n):
        for u in adj[v]:
            if v < u:
                edges.append((v, u))
    if _verify_girth(n, edges, g):
        return edges
    return None


def _find_short_cycle(n, adj, g):
    """Find a cycle shorter than g using BFS from each vertex."""
    for start in range(n):
        # BFS from start
        dist = {start: -1}
        parent = {start: None}
        queue = _deque([start])
        found = False
        while queue and not found:
            node = queue.popleft()
            for nb in adj[node]:
                if nb == parent.get(node):
                    continue
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    if dist[nb] + 1 < g:  # Cycle length = dist[nb] + 1 (back to start via parent chain)
                        # Actually check: path from start to nb has length dist[nb]
                        # If nb is connected to start, cycle length = dist[nb] + 1
                        if start in adj[nb]:
                            # Build cycle
                            cycle = []
                            cur = nb
                            while cur != start:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.append(start)
                            return cycle
                    queue.append(nb)
    return None


def _verify_girth(n, edges, g):
    """Verify girth >= g using non-backtracking walks from every vertex."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Check degree
    for v in range(n):
        if len(adj[v]) != len(set(adj[v])):
            return False  # Multi-edge

    # Non-backtracking walk: no cycle shorter than g
    # BFS with state (vertex, prev_vertex) to track non-backtracking walks
    for start in range(n):
        # State: (current, previous)
        visited = set()
        queue = _deque()
        for nb in adj[start]:
            state = (nb, start)
            if state not in visited:
                visited.add(state)
                queue.append((state, 1))

        while queue:
            (node, prev), dist = queue.popleft()
            if dist >= g - 1:
                continue  # No need to go further
            for nb in adj[node]:
                if nb == prev:
                    continue  # Non-backtracking
                state = (nb, node)
                if state in visited:
                    # Found a cycle of length dist + 1
                    if dist + 1 < g:
                        return False
                    continue
                visited.add(state)
                queue.append((state, dist + 1))

    return True
# EVOLVE-BLOCK-END
