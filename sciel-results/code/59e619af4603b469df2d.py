"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_bicolor_forest(colors, n, edges):
    """Check that for every pair of colors (a,b), the subgraph induced by vertices
    with color a or b is a forest (no cycles). Uses union-find per pair."""
    from collections import defaultdict
    color_vertices = defaultdict(list)
    for i in range(n):
        color_vertices[colors[i]].append(i)

    used_colors = sorted(color_vertices.keys())

    for idx_a in range(len(used_colors)):
        for idx_b in range(idx_a + 1, len(used_colors)):
            a, b = used_colors[idx_a], used_colors[idx_b]
            verts = set(color_vertices[a]) | set(color_vertices[b])
            parent = {v: v for v in verts}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True

            for u, v in edges:
                if u in verts and v in verts:
                    if not union(u, v):
                        return False
    return True


def construct():
    deadline = time.time() + 82.0  # safety margin under 90s wall clock
    n = P['n']
    edges = P['edges']

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Order vertices by degree descending, tie-break by index
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))

    # The incremental validity check.
    # When assigning color c to vertex v:
    #  (1) no colored neighbor may already have color c.
    #  (2) For each distinct neighbor-color d != c, if v has >=2 neighbors of
    #      color d, those neighbors must NOT be connected in the current
    #      subgraph induced by colors {c, d} (otherwise adding v + its edges
    #      to them closes a bicolored cycle).
    # We test connectivity with a bounded BFS restricted to vertices colored
    # c or d, starting from one anchor neighbor.

    def is_valid(color, v, c):
        # (1) properness
        for u in adj[v]:
            if color[u] == c:
                return False

        # (2) bicolored cycle
        # gather distinct neighbor colors (only colored, != c)
        neighbor_colors = set()
        for u in adj[v]:
            d = color[u]
            if d != -1 and d != c:
                neighbor_colors.add(d)

        for d in neighbor_colors:
            # neighbors of v with color d
            neighbors_d = [w for w in adj[v] if color[w] == d]
            if len(neighbors_d) < 2:
                continue
            # BFS from the first anchor through {c,d}-colored vertices;
            # if we can reach a second distinct neighbor_d, a path exists.
            anchor = neighbors_d[0]
            visited = set()
            queue = deque([anchor])
            visited.add(anchor)
            bad = False
            while queue and not bad:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    cn = color[nxt]
                    if cn != c and cn != d:
                        continue
                    if nxt in visited:
                        continue
                    if nxt in neighbors_d and nxt != anchor:
                        bad = True
                        break
                    visited.add(nxt)
                    queue.append(nxt)
            if bad:
                return False
        return True

    def backtrack(idx, color, k):
        if time.time() > deadline:
            return None
        if idx == n:
            return color[:]

        v = order[idx]

        # color ordering: try 0..k-1
        for c in range(k):
            if is_valid(color, v, c):
                color[v] = c
                res = backtrack(idx + 1, color, k)
                if res is not None:
                    return res
                color[v] = -1
        return None

    # Try increasing k; return first valid minimal-color solution.
    for k in range(3, 13):
        if time.time() > deadline - 2.0:
            break
        color = [-1] * n
        res = backtrack(0, color, k)
        if res is not None:
            # Defensive verification (should always pass)
            if _check_bicolor_forest(res, n, edges):
                return [int(x) for x in res]
            # If the incremental check was ever wrong, fall through to fallback
            break

    # Fallback: 50 distinct colors is always valid.
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
