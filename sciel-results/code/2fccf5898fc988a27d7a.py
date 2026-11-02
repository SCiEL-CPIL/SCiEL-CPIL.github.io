import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def check_girth_nb_fast(adj, n, g, start_vertex=0):
        """
        Check girth using non-backtracking walks from a single start vertex.
        For cyclic/vertex-transitive graphs, checking from one vertex is sufficient.
        Returns True if no cycle shorter than g exists starting from start_vertex.
        """
        # We only need to check from start_vertex for vertex-transitive graphs (like cyclic ones).
        # If a short cycle exists, it will be found from any vertex due to symmetry.

        for first_neighbor in adj[start_vertex]:
            # State: (node, parent)
            # We use a list of integers for visited (depth first seen).
            # State index: node * n + parent
            visited = [-1] * (n * n)
            queue = deque()
            queue.append((first_neighbor, start_vertex, 1))
            visited[first_neighbor * n + start_vertex] = 1

            while queue:
                curr, prev, depth = queue.popleft()

                # Check if we can close the cycle to 'start_vertex'
                # Edge (curr, start_vertex) is valid if start_vertex != prev
                if start_vertex != prev:
                    if start_vertex in adj[curr]:
                        cycle_len = depth + 1
                        if cycle_len < g:
                            return False

                # Expand
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    state_idx = nxt * n + curr
                    if visited[state_idx] == 0:
                        visited[state_idx] = 1
                        queue.append((nxt, curr, depth + 1))

                # Pruning: if depth + 1 >= g, we don't need to go deeper
                if depth + 1 >= g:
                    continue

        return True

    def make_cyclic_graph(n, offsets):
        """
        Create a cyclic Z_n-regular graph with given offsets.
        Returns None if invalid.
        """
        edges = []
        edge_set = set()

        if 0 in offsets:
            return None
        if len(set(o % n for o in offsets)) < len(offsets):
            return None

        # Check for duplicate edges or loops
        for i in range(len(offsets)):
            for j in range(i + 1, len(offsets)):
                if (offsets[i] + offsets[j]) % n == 0:
                    return None

        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None
                edge_set.add(key)
                edges.append((i, j))

        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None

        return edges

    def check_girth_from_edges_fast(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # For cyclic graphs, checking from vertex 0 is sufficient
        return check_girth_nb_fast(adj, n, g, 0)

    def random_valid_offsets(n, k):
        """Generate k random valid offsets for a cyclic graph."""
        offsets = []
        valid_candidates = []
        for o in range(1, n):
            # Check if o is valid (not causing loops or duplicate edges)
            # A simple heuristic: avoid small o and o+n/2 type issues
            # But we'll just try random ones and validate
            pass

        # Just try random offsets and check
        while len(offsets) < k:
            o = random.randint(1, n - 1)
            if o in offsets:
                continue
            valid = True
            for existing in offsets:
                if (o + existing) % n == 0:
                    valid = False
                    break
            if valid:
                offsets.append(o)
        return offsets

    best_n = float('inf')
    best_edges = None

    # Strategy:
    # 1. Try n=320 with many random attempts (cyclic graphs).
    # 2. If not found, try n=322, 324, etc.
    # 3. If not found, fallback to LW graph.

    search_ns = [320] + list(range(322, 350, 2)) + list(range(243, 320, 10))

    for n in search_ns:
        if time.time() > deadline - 5:
            break

        if n == 320:
            attempts = 50000
        elif n < 320:
            attempts = 200
        else:
            attempts = 5000

        for _ in range(attempts):
            if time.time() > deadline - 2:
                break

            offsets = random_valid_offsets(n, k)
            edges = make_cyclic_graph(n, offsets)
            if edges is not None:
                if check_girth_from_edges_fast(edges, n, g):
                    return n, edges

    # Fallback to LW graph
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
