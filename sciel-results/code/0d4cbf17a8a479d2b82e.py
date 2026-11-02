import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 20s margin before 120s limit

    # Moore bound: n >= k + k(k-1) + ... + k(k-1)^(d-1) where d = floor((g-1)/2)
    # For girth g, the graph is tree-like up to depth d = floor((g-1)/2)
    # Actually for girth g, BFS from a vertex to depth (g-1)/2 is tree-like
    # Moore bound for k-regular graph of girth g:
    # n >= 1 + k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^((g-1)/2 - 1)) if g is odd
    # More precisely, for girth g (odd): n >= 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i
    # For girth g (even): n >= 2k * sum_{i=0}^{g/2 - 1} (k-1)^i

    def moore_bound(k, g):
        if g % 2 == 1:
            d = (g - 1) // 2  # depth from center
            # 1 + k * sum_{i=0}^{d-1} (k-1)^i
            s = 0
            for i in range(d):
                s += (k - 1) ** i
            return 1 + k * s
        else:
            d = g // 2 - 1  # sum goes to d
            s = 0
            for i in range(d + 1):
                s += (k - 1) ** i
            return 2 * k * s

    mb = moore_bound(k, g)

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def construct_luw(k, g):
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

    # Try to construct a smaller graph using random regular graph + girth check
    # We'll try to find a k-regular graph with girth >= g on n vertices
    # where n is between moore_bound and some upper bound

    def check_girth(adj, n, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        depth = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        # But we need to be careful: this might not be the shortest
                        # If dist[u] + dist[v] + 1 < min_girth, then girth < min_girth
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
            # Also check for self-loops and back edges
            # Actually the BFS above should catch cycles
        return True

    def generate_random_regular_graph(n, k):
        """Generate a random k-regular graph using configuration model."""
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        if k == n - 1:
            # Complete graph
            adj = [set(range(n)) - {i} for i in range(n)]
            return adj
        if k == 0:
            return [set() for _ in range(n)]
        if k == 1:
            if n % 2 != 0:
                return None
            adj = [set() for _ in range(n)]
            for i in range(n // 2):
                u, v = 2 * i, 2 * i + 1
                adj[u].add(v)
                adj[v].add(u)
            return adj

        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)

        adj = [set() for _ in range(n)]
        i = 0
        while i < len(stubs) - 1:
            u = stubs[i]
            v = stubs[i + 1]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                i += 2
            else:
                # Find another valid partner
                found = False
                for j in range(i + 2, len(stubs)):
                    v2 = stubs[j]
                    if u != v2 and v2 not in adj[u]:
                        adj[u].add(v2)
                        adj[v2].add(u)
                        # Swap to remove v2 from its position
                        stubs[j], stubs[i + 1] = stubs[i + 1], stubs[j]
                        i += 2
                        found = True
                        break
                if not found:
                    return None
        # Verify regularity
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return adj

    def find_smallest_graph(max_n, time_limit):
        """Try to find the smallest k-regular graph with girth >= g."""
        best_n = None
        best_adj = None

        # Try n values from moore_bound upward
        for n in range(mb, min(mb + 200, max_n + 1)):
            if time.time() > time_limit:
                break
            # Try several random instances
            for attempt in range(20):
                if time.time() > time_limit:
                    break
                adj = generate_random_regular_graph(n, k)
                if adj is None:
                    continue
                if check_girth(adj, n, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_adj = adj
                    break  # Found one at this n, move to next n

        if best_adj is not None:
            edges = []
            for u in range(best_n):
                for v in best_adj[u]:
                    if u < v:
                        edges.append((u, v))
            return best_n, edges
        return None

    # First try the random approach with a reasonable time budget
    try_budget = 60  # seconds
    result = find_smallest_graph(1000, start_time + try_budget)

    if result is not None:
        return result

    # Fallback to LUW
    return construct_luw(k, g)
