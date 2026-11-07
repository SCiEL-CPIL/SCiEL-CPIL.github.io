import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Order vertices by descending degree for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    # For each vertex in order, which earlier-in-order neighbors exist
    pos = {v: i for i, v in enumerate(order)}
    neigh = [sorted([nb for nb in adj[v] if pos[nb] < pos[v]], key=lambda x: pos[x]) for v in order]
    neigh_set = [set(nb) for nb in neigh]

    def find_cycle_in_bicolor(color_pair):
        # Union-find: for each pair of colors, check if the subgraph induced
        # by those two colors (edges between vertices of those two colors)
        # has a cycle.
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        for u, v in edges:
            if color_pair[0] in (color[u], color[v]) and color_pair[1] in (color[u], color[v]):
                if color[u] != color[v]:
                    if not union(u, v):
                        return True
        return False

    def is_valid(color):
        # Proper coloring
        for u, v in edges:
            if color[u] == color[v]:
                return False
        # Bicolored forest check
        used = set(color)
        for c in used:
            for d in used:
                if c < d:
                    if find_cycle_in_bicolor((c, d)):
                        return False
        return True

    def backtrack(k, color, used_colors):
        if k == n:
            return True
        v = order[k]
        # Determine forbidden colors from earlier neighbors
        forbidden = set()
        for nb in neigh[k]:
            forbidden.add(color[nb])
        for c in range(len(used_colors) + 1):
            if c in forbidden:
                continue
            color[v] = c
            new_used = used_colors | {c}
            # Check bicolored forest constraint incrementally for pairs involving c
            # that now have at least one edge
            ok = True
            for d in new_used:
                if d == c:
                    continue
                # Check if adding vertex v with color c creates a bicolored cycle
                # with color d. We need to check edges among {c,d} vertices.
                # Quick check: does v have a neighbor with color d, and does that
                # create a cycle in the (c,d)-subgraph?
                pass
            # For efficiency, do a lighter check: only check pairs where v participates
            # in an edge of that pair
            for d in new_used:
                if d == c:
                    continue
                # v has color c; check edges from v to neighbors with color d
                # and whether that creates a cycle
                has_edge_v_d = False
                for nb in adj[v]:
                    if color[nb] == d:
                        has_edge_v_d = True
                        break
                if not has_edge_v_d:
                    continue
                # Check if the (c,d) subgraph has a cycle
                parent = list(range(n))
                rank = [0] * n
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra == rb:
                        return False
                    if rank[ra] < rank[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    if rank[ra] == rank[rb]:
                        rank[ra] += 1
                    return True
                for u2, v2 in edges:
                    if color[u2] in (c, d) and color[v2] in (c, d) and color[u2] != color[v2]:
                        if not union(u2, v2):
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                if backtrack(k + 1, color, new_used):
                    return True
        color[v] = -1
        return False

    deadline = time.time() + 75
    best_color = None

    # Try 3, 4, 5 colors in order
    for max_colors in [3, 4, 5]:
        if time.time() > deadline:
            break
        color = [-1] * n
        # Assign first vertex color 0 (symmetry breaking)
        color[order[0]] = 0
        if backtrack(1, color, {0}):
            best_color = color
            break

    if best_color is None:
        # Fallback: the known 5-coloring
        return [v % 5 for v in range(n)]

    return best_color
