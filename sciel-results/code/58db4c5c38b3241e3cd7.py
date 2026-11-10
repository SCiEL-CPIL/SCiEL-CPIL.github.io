"""Simulated annealing for bicolored-forest coloring of C(50;{1,7})."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_bicolored_forest(colors, edges, c1, c2):
    """Check that subgraph induced by colors c1 and c2 is a forest using union-find."""
    vertices = set()
    for i, c in enumerate(colors):
        if c == c1 or c == c2:
            vertices.add(i)
    if len(vertices) <= 1:
        return True

    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    for u, v in edges:
        if u in vertices and v in vertices:
            if not union(u, v):
                return False
    return True


def _is_valid(colors, edges):
    """Full validity check: proper coloring + all bicolored subgraphs are forests."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    used = set(colors)
    for i, c1 in enumerate(used):
        for c2 in used:
            if c1 >= c2:
                continue
            if not _check_bicolored_forest(colors, edges, c1, c2):
                return False
    return True


def _count_violations(colors, edges, adj, n):
    """Count total violations: adjacent same-color + bicolored cycles."""
    violations = 0
    # Count adjacent same-color violations
    for u, v in edges:
        if colors[u] == colors[v]:
            violations += 1

    # Count bicolored cycle violations
    used = set(colors)
    for i, c1 in enumerate(used):
        for c2 in used:
            if c1 >= c2:
                continue
            if not _check_bicolored_forest(colors, edges, c1, c2):
                violations += 1
    return violations


def _fast_bicolored_check(colors, edges, c1, c2, n):
    """Fast check if bicolored subgraph for c1,c2 has a cycle."""
    # Collect vertices with color c1 or c2
    in_set = [False] * n
    for i in range(n):
        if colors[i] == c1 or colors[i] == c2:
            in_set[i] = True

    # Union-find with path compression
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    for u, v in edges:
        if in_set[u] and in_set[v]:
            if not union(u, v):
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for 90s wall-clock limit
    random.seed(42)

    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Try to find a valid coloring with few colors using simulated annealing
    best_colors = None
    best_num_colors = n + 1

    # Try different numbers of colors from low to high
    for num_colors in range(3, 15):
        if time.time() > deadline - 5:
            break

        # Multiple restarts for each num_colors
        max_restarts = 50 if num_colors <= 5 else 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            # Initialize: random proper-ish coloring
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]

            # Improve initial solution with local search
            # Simulated annealing
            T = 10.0
            cooling = 0.999
            min_T = 0.01
            current_cost = _count_violations(colors, edges, adj, n)
            best_local = list(colors)
            best_local_cost = current_cost

            steps = 0
            max_steps = 50000
            while T > min_T and steps < max_steps:
                if time.time() > deadline - 5:
                    break
                steps += 1

                # Pick a random vertex to change
                v = random.randint(0, n - 1)
                old_c = colors[v]
                new_c = random.randint(0, num_colors - 1)
                if new_c == old_c:
                    continue

                # Check if new_c conflicts with neighbors
                neighbor_colors = set(colors[nb] for nb in adj[v])
                if new_c in neighbor_colors:
                    # Still might be worth trying if it reduces other violations
                    pass

                colors[v] = new_c
                new_cost = _count_violations(colors, edges, adj, n)
                delta = new_cost - current_cost

                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    current_cost = new_cost
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local = list(colors)
                else:
                    colors[v] = old_c

                T *= cooling

            if best_local_cost == 0:
                # Valid solution found!
                # Count distinct colors used
                distinct = len(set(best_local))
                if distinct < best_num_colors:
                    best_num_colors = distinct
                    best_colors = best_local
                break  # Found valid for this num_colors, try next

    # If we found a valid solution, return it
    if best_colors is not None:
        return [int(c) for c in best_colors]

    # Fallback: try a structured approach with more colors
    # Use 5 colors with a pattern based on the circulant structure
    # C(50;{1,7}): edges (i,i+1) and (i,i+7) mod 50
    # Try: color(i) = (i % 5)
    for pattern_func in [
        lambda i: i % 5,
        lambda i: (i * 2) % 5,
        lambda i: (i * 3) % 5,
        lambda i: (i * 4) % 5,
        lambda i: (i // 10) % 5,
        lambda i: (i % 10) // 2,
        lambda i: ((i % 25) // 5),
        lambda i: (i % 7),
        lambda i: (i * 3) % 7,
    ]:
        if time.time() > deadline - 2:
            break
        colors = [pattern_func(i) for i in range(n)]
        if _is_valid(colors, edges):
            return [int(c) for c in colors]

    # Last resort: greedy with 10 colors
    colors = [None] * n
    for v in range(n):
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] is not None:
                forbidden.add(colors[nb])
        for c in range(10):
            if c not in forbidden:
                colors[v] = c
                break
        if colors[v] is None:
            colors[v] = 0

    return [int(c) for c in colors]
