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

    # For each color pair (c1, c2), we need the subgraph induced by vertices
    # colored c1 or c2 to be a forest. This means no cycles can be formed
    # using only edges where both endpoints have colors in {c1, c2}.

    # We'll use backtracking with forward checking.
    # Key insight: for a fixed set of k colors, we need to assign each vertex
    # one of the k colors such that:
    # 1. Adjacent vertices have different colors
    # 2. For every pair of colors (a, b), the subgraph induced by vertices
    #    with colors a or b is a forest (acyclic)

    # We'll try k = 3, 4, 5, ... until we find a valid coloring.

    def solve_with_k_colors(k):
        """Try to find a valid coloring using exactly k colors (0..k-1)."""
        color = [-1] * n

        # DSU for each color pair to detect cycles
        # pair_dsu[(a,b)] where a < b, maps to parent array
        # We'll track edges in each pair's subgraph

        # For efficiency, we track for each color pair the DSU
        def make_dsu():
            return list(range(n))

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

        # State: for each color pair (a,b) with a<b, we maintain a DSU
        # and the set of edges in that pair's subgraph
        # When we assign color c to vertex v, for each neighbor u already colored c' != c:
        #   the pair (min(c,c'), max(c,c')) gets a new edge (v,u)
        #   We check if this creates a cycle in that pair's DSU

        # Also, when we assign color c to vertex v, for each other vertex w already colored c:
        #   No direct edge constraint (they're same color, not adjacent by rule 1)
        #   But wait - same color vertices aren't adjacent (rule 1), so no edge between them

        # Actually, the bicolored subgraph for pair (a,b) includes ALL edges where
        # both endpoints have colors in {a,b}. So if vertex v has color a and vertex u
        # has color b, and (v,u) is an edge, then (v,u) is in the subgraph for pair (a,b).
        # Also if both have color a, but they can't be adjacent (rule 1).
        # Similarly both color b can't be adjacent.

        # So for pair (a,b), the subgraph has edges only between vertices colored a and vertices colored b.
        # It's a bipartite graph! And we need it to be a forest (no cycles).

        # This is a crucial simplification: the bicolored subgraph for (a,b) is bipartite
        # with parts being vertices colored a and vertices colored b.

        # Let's use backtracking with the following state:
        # - color[v] for each vertex
        # - For each pair (a,b), a DSU to detect if adding an edge creates a cycle

        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        pos_in_order = [0] * n
        for i, v in enumerate(order):
            pos_in_order[v] = i

        # We'll process vertices in this order
        # For forward checking: when assigning color c to vertex v, check all uncolored neighbors

        best_solution = None

        def backtrack(idx):
            nonlocal best_solution
            if time.time() > deadline:
                return False

            if idx == n:
                best_solution = color[:]
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

                # Check: for each already-colored neighbor u with color cu != c,
                # adding edge (v,u) to pair (min(c,cu), max(c,cu)) must not create a cycle
                conflict = False
                added_edges = []  # list of (pair, v, u) to undo

                for u in adj[v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        a, b = (c, cu) if c < cu else (cu, c)
                        # Check if adding (v,u) creates a cycle in pair (a,b)'s subgraph
                        # In the DSU for pair (a,b), check if v and u are already connected
                        dsu = pair_dsus[(a, b)]
                        if find(dsu, v) == find(dsu, u):
                            conflict = True
                            break
                        # Tentatively union
                        union(dsu, v, u)
                        added_edges.append((a, b, v, u))

                if conflict:
                    # Undo
                    for a, b, vv, uu in added_edges:
                        # We need to undo the union - but DSU doesn't support undo easily
                        # Instead, let's use a different approach: save state
                        pass
                    continue

                # Assign
                color[v] = c
                if backtrack(idx + 1):
                    return True

                # Undo
                color[v] = -1
                for a, b, vv, uu in added_edges:
                    # Need to undo - this is problematic with DSU
                    pass

            return False

        # The DSU undo problem is tricky. Let's use a different approach:
        # Instead of maintaining DSUs incrementally, we can check for cycles
        # by doing a BFS/DFS on the bicolored subgraph when needed.
        # But that's expensive.

        # Alternative: use the fact that the bicolored subgraph is bipartite.
        # A bipartite graph is a forest iff it has no cycles iff
        # for each connected component, edges = vertices - 1.
        # But tracking this incrementally is also complex.

        # Let me try a simpler approach: just do full validation at each step
        # but with good vertex ordering and color ordering.

        # Actually, let's try a cleaner backtracking with proper state saving.
        # For each pair (a,b), maintain:
        #   - number of edges in the subgraph
        #   - DSU (with path compression disabled for undo, or use union by rank without compression)
        #   - or simply: the list of edges, and check acyclicity via DFS when needed

        # Given n=50 and small k, let's just do:
        # At each assignment, validate all affected pairs by checking if their
        # induced subgraph is a forest using DFS.

        # For pair (a,b), the subgraph has vertices V_ab = {v : color[v] in {a,b}}
        # and edges E_ab = {(u,v) in edges : color[u] in {a,b} and color[v] in {a,b}}
        # This is a forest iff it's acyclic.

        # Let's implement this check efficiently.

        def is_forest_for_pair(a, b, color_arr):
            """Check if subgraph induced by colors a and b is a forest."""
            # Collect vertices with color a or b
            verts = set()
            for v in range(n):
                if color_arr[v] == a or color_arr[v] == b:
                    verts.add(v)

            if len(verts) <= 1:
                return True

            # Build adjacency within this subgraph
            # Edge (u,v) is in subgraph if both u,v in verts and (u,v) in edges
            # Use BFS/DFS to check for cycles
            visited = set()
            for start in verts:
                if start not in visited:
                    # BFS
                    queue = deque([start])
                    visited.add(start)
                    edge_count = 0
                    vert_count = 0
                    while queue:
                        node = queue.popleft()
                        vert_count += 1
                        for nb in adj[node]:
                            if nb in verts:
                                edge_count += 1  # count each edge twice
                                if nb not in visited:
                                    visited.add(nb)
                                    queue.append(nb)
                    # edge_count is doubled, so actual edges = edge_count // 2
                    # For a tree: edges = vertices - 1
                    if edge_count // 2 != vert_count - 1:
                        return False
            return True

        # This is O(n) per check, and we do it for O(k^2) pairs, so O(n*k^2) per step.
        # With n=50, k<=5, this is manageable.

        color = [-1] * n
        solution = None

        def bt(idx):
            nonlocal solution
            if time.time() > deadline:
                return False

            if idx == n:
                solution = color[:]
                return True

            v = order[idx]

            # Try colors in order
            for c in range(k):
                # Check neighbor constraint
                ok = True
                for u in adj[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue

                # Tentatively assign
                color[v] = c

                # Check all pairs involving color c
                valid = True
                for other in range(k):
                    if other == c:
                        continue
                    if not is_forest_for_pair(c, other, color):
                        valid = False
                        break

                if valid:
                    if bt(idx + 1):
                        return True

                color[v] = -1

            return False

        if bt(0):
            return solution
        return None

    # Try increasing k
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = solve_with_k_colors(k)
        if result is not None:
            # Verify
            valid = True
            for u, v in edges:
                if result[u] == result[v]:
                    valid = False
                    break
            if valid:
                # Check all color pairs
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
                            # BFS to check acyclicity
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

    # Fallback: use more colors
    # Try k=5 with different vertex ordering
    for k in range(5, 15):
        if time.time() > deadline:
            break
        # Simple greedy
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
                color[v] = k  # shouldn't happen

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
