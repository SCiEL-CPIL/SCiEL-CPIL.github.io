"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def find_bicolored_cycles(coloring):
        """Find all bicolored cycles. Returns set of edges involved in cycles."""
        colors_used = sorted(set(coloring))
        cycle_edges = set()
        for i in range(len(colors_used)):
            for j in range(i + 1, len(colors_used)):
                a, b = colors_used[i], colors_used[j]
                verts = set()
                for v in range(n):
                    if coloring[v] == a or coloring[v] == b:
                        verts.add(v)
                if len(verts) < 3:
                    continue
                # Find cycles in this subgraph using DFS
                visited = set()
                for start in verts:
                    if start in visited:
                        continue
                    # DFS to find cycle
                    stack = [(start, start, 0)]
                    visited_local = set()
                    while stack:
                        node, parent, depth = stack.pop()
                        if node in visited_local:
                            continue
                        visited_local.add(node)
                        for nb in adj[node]:
                            if nb not in verts:
                                continue
                            if nb == parent:
                                continue
                            if nb in visited_local:
                                # Found a cycle - mark edges
                                # We need to find the actual cycle path
                                # For cost purposes, just count this as a violation
                                cycle_edges.add((min(node, nb), max(node, nb)))
                                cycle_edges.add((min(node, parent), max(node, parent)))
                            else:
                                stack.append((nb, node, depth + 1))
                    visited.update(visited_local)
        return cycle_edges

    def count_bicolored_cycles(coloring):
        """Count number of bicolored cycles (for cost function)."""
        colors_used = sorted(set(coloring))
        total_cycles = 0
        for i in range(len(colors_used)):
            for j in range(i + 1, len(colors_used)):
                a, b = colors_used[i], colors_used[j]
                verts = set()
                for v in range(n):
                    if coloring[v] == a or coloring[v] == b:
                        verts.add(v)
                if len(verts) < 3:
                    continue
                # Count cycles: for each connected component, cycles = E - V + 1
                visited = set()
                for start in verts:
                    if start in visited:
                        continue
                    queue = deque([start])
                    visited.add(start)
                    vc = 0
                    ec = 0
                    while queue:
                        node = queue.popleft()
                        vc += 1
                        for nb in adj[node]:
                            if nb in verts:
                                ec += 1
                                if nb not in visited:
                                    visited.add(nb)
                                    queue.append(nb)
                    ec //= 2
                    if vc > 1:
                        total_cycles += max(0, ec - vc + 1)
        return total_cycles

    def count_proper_violations(coloring):
        """Count adjacent vertices with same color."""
        violations = 0
        for u, v in edges:
            if coloring[u] == coloring[v]:
                violations += 1
        return violations

    def total_cost(coloring):
        """Total cost: proper violations weighted heavily + bicolored cycles."""
        pv = count_proper_violations(coloring)
        bc = count_bicolored_cycles(coloring)
        return pv * 1000 + bc

    def is_valid(coloring):
        """Check if coloring is fully valid."""
        if count_proper_violations(coloring) > 0:
            return False
        if count_bicolored_cycles(coloring) > 0:
            return False
        return True

    def local_search(k, max_iter=50000, temp_start=1.0, temp_end=0.01):
        """Simulated annealing local search."""
        # Initialize with random coloring
        color = [random.randint(0, k - 1) for _ in range(n)]
        current_cost = total_cost(color)
        best_color = color[:]
        best_cost = current_cost

        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Pick random vertex and random new color
            v = random.randint(0, n - 1)
            old_c = color[v]
            new_c = random.randint(0, k - 1)
            if new_c == old_c:
                continue
            color[v] = new_c
            new_cost = total_cost(color)
            delta = new_cost - current_cost
            temp = temp_start * (temp_end / temp_start) ** (it / max_iter)
            if delta < 0 or random.random() < (2.718281828 ** (-delta / max(temp, 1e-10))):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_color = color[:]
            else:
                color[v] = old_c

        return best_color, best_cost

    def greedy_init(k):
        """Greedy initialization."""
        color = [-1] * n
        order = list(range(n))
        random.shuffle(order)
        for v in order:
            used = set()
            for u in adj[v]:
                if color[u] >= 0:
                    used.add(color[u])
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            else:
                color[v] = 0
        return color

    # Try to find solution with 4 colors first (most likely minimum)
    best_result = None
    best_num_colors = 50

    for k in [4, 5, 6]:
        if time.time() > deadline:
            break

        # Multiple restarts of local search
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break

            # Greedy init then local search
            color = greedy_init(k)
            color, cost = local_search(k, max_iter=30000)

            if cost == 0:
                if is_valid(color):
                    num_colors = len(set(color))
                    if num_colors < best_num_colors:
                        best_num_colors = num_colors
                        best_result = color[:]
                    found = True
                    break

        if found:
            break

    # If no valid solution found, use fallback
    if best_result is None:
        # Try with more colors
        for k in range(7, 12):
            if time.time() > deadline:
                break
            color = greedy_init(k)
            color, cost = local_search(k, max_iter=20000)
            if cost == 0 and is_valid(color):
                num_colors = len(set(color))
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_result = color[:]
                break

    if best_result is not None:
        # Ensure colors are in range 0..49 and are Python ints
        result = [int(c) for c in best_result]
        # Remap to smallest possible labels
        unique_colors = sorted(set(result))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        result = [mapping[c] for c in result]
        return result

    # Last resort: trivial solution
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
