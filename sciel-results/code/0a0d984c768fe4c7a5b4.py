"""Rewritten program for few-colors-without-bicolored-cycle task."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s

    n = P['n']
    edges = P['edges']

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Precompute edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    def check_bicolored_cycle(coloring, c1, c2):
        """Check if subgraph induced by colors c1 and c2 has a cycle using union-find."""
        vertices = [i for i in range(n) if coloring[i] in (c1, c2)]
        if len(vertices) < 3:
            return False

        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        for u in vertices:
            for v in adj[u]:
                if v > u and coloring[v] in (c1, c2):
                    if not union(u, v):
                        return True
        return False

    def is_valid(coloring):
        """Check proper coloring and no bicolored cycles."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False

        colors_used = set(coloring)
        for i, c1 in enumerate(colors_used):
            for c2 in list(colors_used)[i+1:]:
                if check_bicolored_cycle(coloring, c1, c2):
                    return False
        return True

    def incremental_cycle_check(coloring, v, new_color, old_color):
        """
        After changing vertex v from old_color to new_color,
        check if any bicolored subgraph involving new_color has a cycle.
        Only need to check pairs involving new_color (and old_color if it was the only one).
        """
        # Check pairs involving new_color
        for c in range(50):
            if c == new_color:
                continue
            # Only check if there's at least one vertex with color c
            if not any(coloring[i] == c for i in range(n)):
                continue
            if check_bicolored_cycle(coloring, new_color, c):
                return False
        return True

    def backtrack_with_cycle_check(num_colors, order):
        """
        Backtracking with incremental bicolored cycle detection.
        order: list of vertices in assignment order
        """
        coloring = [-1] * n
        # Track which colors are used
        colors_in_use = [False] * num_colors

        def has_cycle_with_color(c1, c2):
            """Quick check: does the current partial coloring have a bicolored cycle for colors c1, c2?"""
            vertices = [i for i in range(n) if coloring[i] in (c1, c2)]
            if len(vertices) < 3:
                return False

            parent = {v: v for v in vertices}
            rank = {v: 0 for v in vertices}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True

            for u in vertices:
                for v in adj[u]:
                    if v > u and coloring[v] in (c1, c2):
                        if not union(u, v):
                            return True
            return False

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True

            v = order[idx]
            for c in range(num_colors):
                # Check proper coloring constraint
                allowed = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        allowed = False
                        break
                if not allowed:
                    continue

                coloring[v] = c
                colors_in_use[c] = True

                # Check bicolored cycles involving color c
                valid = True
                for c2 in range(num_colors):
                    if c2 == c or not colors_in_use[c2]:
                        continue
                    if has_cycle_with_color(c, c2):
                        valid = False
                        break

                if valid and backtrack(idx + 1):
                    return True

                coloring[v] = -1
                colors_in_use[c] = False
                # Re-check if c is still in use by other vertices
                if any(coloring[i] == c for i in range(n)):
                    colors_in_use[c] = True

            return False

        if backtrack(0):
            return coloring[:]
        return None

    def greedy_init(num_colors, seed=None):
        """Greedy coloring with deterministic or random ordering."""
        coloring = [-1] * n
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        order = list(range(n))
        rng.shuffle(order)

        for v in order:
            used = set(coloring[nb] for nb in adj[v] if coloring[nb] != -1)
            available = [c for c in range(num_colors) if c not in used]
            if available:
                coloring[v] = rng.choice(available)
            else:
                coloring[v] = 0
        return coloring

    def local_search(coloring, num_colors, max_iters=3000):
        """Local search minimizing violations."""
        def count_violations(col):
            violations = 0
            for u, v in edges:
                if col[u] == col[v]:
                    violations += 1
            colors_used = set(col)
            for i, c1 in enumerate(colors_used):
                for c2 in list(colors_used)[i+1:]:
                    if check_bicolored_cycle(col, c1, c2):
                        violations += 1
            return violations

        current_violations = count_violations(coloring)
        best = coloring[:]
        best_violations = current_violations

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_violations == 0:
                break

            v = random.randint(0, n - 1)
            old_color = coloring[v]
            best_new = old_color
            best_new_viol = current_violations

            for nc in range(num_colors):
                if nc == old_color:
                    continue
                if any(coloring[nb] == nc for nb in adj[v]):
                    continue
                coloring[v] = nc
                nv = count_violations(coloring)
                if nv < best_new_viol:
                    best_new_viol = nv
                    best_new = nc

            if best_new != old_color:
                coloring[v] = best_new
                current_violations = best_new_viol
                if current_violations < best_violations:
                    best_violations = current_violations
                    best = coloring[:]
            else:
                coloring[v] = old_color

        return best, best_violations

    # Strategy 1: Try backtracking with 3 colors first
    # Order vertices by degree descending for better pruning
    order_by_degree = sorted(range(n), key=lambda x: -len(adj[x]))

    for nc in [3, 4, 5, 6]:
        if time.time() > deadline:
            break

        # Try backtracking
        result = backtrack_with_cycle_check(nc, order_by_degree)
        if result is not None and is_valid(result):
            return [int(c) for c in result]

        # Try local search with multiple restarts
        for restart in range(30):
            if time.time() > deadline:
                break
            coloring = greedy_init(nc, seed=restart * 1000 + 42)
            coloring, violations = local_search(coloring, nc, max_iters=2000)
            if violations == 0 and is_valid(coloring):
                return [int(c) for c in coloring]

    # Fallback: all distinct colors
    return [int(c) for c in range(n)]
# EVOLVE-BLOCK-END
