"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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

    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i

    # For each color pair (a, b) with a < b, we maintain a DSU to track
    # whether adding an edge between vertices colored a and b would create a cycle.
    # The bicolored subgraph for pair (a,b) is bipartite (vertices colored a vs b),
    # and we need it to be a forest.

    def solve_with_k_colors(k):
        """Try to find a valid coloring using exactly k colors (0..k-1)."""
        color = [-1] * n
        solution = None

        # DSU for each color pair
        # pair_dsu[(a,b)] where a < b
        # We'll use a dict mapping (a,b) -> list of parent arrays
        # Actually, let's use a 2D array: dsu[a][b] for a < b
        # Each is a list of n elements (parent pointers)

        # Initialize DSUs
        # For efficiency, we'll use a flat structure
        # dsu[a][b] is a list of n parents for pair (a,b) where a < b
        dsu = {}
        for a in range(k):
            for b in range(a + 1, k):
                dsu[(a, b)] = list(range(n))

        def find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(parent, a, b):
            ra = find(parent, a)
            rb = find(parent, b)
            if ra == rb:
                return False  # cycle detected
            parent[ra] = rb
            return True

        # State to track which unions have been done for backtracking
        # We need to be able to undo unions. Since DSU with path compression
        # is hard to undo, let's use a different approach:
        # Keep a log of all unions performed, and undo them in reverse order.

        union_log = []  # list of (pair_key, ra, rb) where parent[ra] was set to rb

        def do_union(pair_key, u, v):
            """Perform union, return True if successful (no cycle)."""
            parent = dsu[pair_key]
            ra = find(parent, u)
            rb = find(parent, v)
            if ra == rb:
                return False
            parent[ra] = rb
            union_log.append((pair_key, ra, rb))
            return True

        def undo_unions(count):
            """Undo the last 'count' unions."""
            for _ in range(count):
                pair_key, ra, rb = union_log.pop()
                dsu[pair_key][ra] = ra  # restore parent pointer

        def bt(idx):
            nonlocal solution
            if time.time() > deadline:
                return False

            if idx == n:
                solution = color[:]
                return True

            v = order[idx]

            # Try each color
            for c in range(k):
                # Check: no neighbor has color c
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                if not valid:
                    continue

                # Tentatively assign color c to v
                color[v] = c

                # For each already-colored neighbor u with color cu != c,
                # check if adding edge (v,u) to pair (min(c,cu), max(c,cu)) creates a cycle
                conflict = False
                unions_done = 0

                for u in adj[v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        a, b = (c, cu) if c < cu else (cu, c)
                        pair_key = (a, b)
                        if do_union(pair_key, v, u):
                            unions_done += 1
                        else:
                            conflict = True
                            break

                if conflict:
                    # Undo all unions done in this attempt
                    undo_unions(unions_done)
                    color[v] = -1
                    continue

                # Recurse
                if bt(idx + 1):
                    return True

                # Backtrack: undo
                undo_unions(unions_done)
                color[v] = -1

            return False

        if bt(0):
            return solution
        return None

    # Try increasing k from 3
    for k in range(3, 8):
        if time.time() > deadline:
            break
        result = solve_with_k_colors(k)
        if result is not None:
            # Verify the solution
            valid = True
            for u, v in edges:
                if result[u] == result[v]:
                    valid = False
                    break
            if valid:
                # Check all color pairs for bicolored cycles
                colors_used = set(result)
                for i, a in enumerate(colors_used):
                    for b in colors_used:
                        if a >= b:
                            continue
                        # Check if subgraph induced by a,b is a forest
                        verts = set()
                        for v in range(n):
                            if result[v] == a or result[v] == b:
                                verts.add(v)
                        if len(verts) > 1:
                            visited = set()
                            is_tree = True
                            for start in verts:
                                if start not in visited:
                                    queue = deque([start])
                                    visited.add(start)
                                    ec = 0
                                    vc = 0
                                    while queue:
                                        node = queue.popleft()
                                        vc += 1
                                        for nb in adj[node]:
                                            if nb in verts:
                                                ec += 1
                                                if nb not in visited:
                                                    visited.add(nb)
                                                    queue.append(nb)
                                    if ec // 2 != vc - 1:
                                        is_tree = False
                                        break
                            if not is_tree:
                                valid = False
                                break
                    if not valid:
                        break
            if valid:
                return [int(x) for x in result]

    # Fallback: greedy with more colors
    for k in range(5, 15):
        if time.time() > deadline:
            break
        color = [-1] * n
        for v in range(n):
            used = set()
            for u in adj[v]:
                if color[u] >= 0:
                    used.add(color[u])
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            else:
                color[v] = k

        # Verify
        valid = True
        for u, v in edges:
            if color[u] == color[v]:
                valid = False
                break
        if valid:
            colors_used = set(color)
            for a in colors_used:
                for b in colors_used:
                    if a >= b:
                        continue
                    verts = set()
                    for v in range(n):
                        if color[v] == a or color[v] == b:
                            verts.add(v)
                    if len(verts) > 1:
                        visited = set()
                        is_tree = True
                        for start in verts:
                            if start not in visited:
                                queue = deque([start])
                                visited.add(start)
                                ec = 0
                                vc = 0
                                while queue:
                                    node = queue.popleft()
                                    vc += 1
                                    for nb in adj[node]:
                                        if nb in verts:
                                            ec += 1
                                            if nb not in visited:
                                                visited.add(nb)
                                                queue.append(nb)
                                if ec // 2 != vc - 1:
                                    is_tree = False
                                    break
                        if not is_tree:
                            valid = False
                            break
                if not valid:
                    break
        if valid:
            return [int(x) for x in color]

    # Last resort: trivial solution
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
