"""Rewritten program: CSP with MRV, forward checking, and incremental bicolored-forest DSU."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
ADJ = [[] for _ in range(N)]
for a, b in EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)

def _validate(coloring):
    """Verify coloring satisfies all constraints. Returns True if valid."""
    k = max(coloring) + 1
    # Adjacency constraint
    for a, b in EDGES:
        if coloring[a] == coloring[b]:
            return False
    # Bicolored forest constraint
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Build subgraph induced by vertices with color c1 or c2
            verts = set()
            for v in range(N):
                if coloring[v] in (c1, c2):
                    verts.add(v)
            if len(verts) <= 1:
                continue
            # Count edges in induced subgraph
            e_count = 0
            adj_sub = defaultdict(list)
            for a, b in EDGES:
                if coloring[a] in (c1, c2) and coloring[b] in (c1, c2):
                    e_count += 1
                    adj_sub[a].append(b)
                    adj_sub[b].append(a)
            # Count components
            visited = set()
            comps = 0
            for v in verts:
                if v not in visited:
                    comps += 1
                    stack = [v]
                    visited.add(v)
                    while stack:
                        x = stack.pop()
                        for y in adj_sub[x]:
                            if y not in visited:
                                visited.add(y)
                                stack.append(y)
            if e_count > len(verts) - comps:
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    # Search for smallest k
    for k in range(2, 10):
        if time.time() > deadline:
            break

        # Precompute pair index mapping
        pair_map = {}
        npairs = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                pair_map[(c1, c2)] = npairs
                npairs += 1

        # DSU arrays: parent[pi][v], rank[pi][v]
        parent = [[v for v in range(N)] for _ in range(npairs)]
        rank = [[0] * N for _ in range(npairs)]
        color = [-1] * N
        union_stack = []  # (pi, small_root, large_root, old_rank_large)

        def find(pi, v):
            while parent[pi][v] != v:
                v = parent[pi][v]
            return v

        def do_union(pi, a, b):
            ra = find(pi, a)
            rb = find(pi, b)
            if ra == rb:
                return False
            if rank[pi][ra] > rank[pi][rb]:
                ra, rb = rb, ra
            old_rank_rb = rank[pi][rb]
            parent[pi][ra] = rb
            if rank[pi][ra] == rank[pi][rb]:
                rank[pi][rb] += 1
            union_stack.append((pi, ra, rb, old_rank_rb))
            return True

        def undo_union():
            pi, ra, rb, old_rank_rb = union_stack.pop()
            parent[pi][ra] = ra
            rank[pi][rb] = old_rank_rb

        # Vertex ordering: use a fixed order that tends to work well for grid-like graphs
        # Try BFS order from vertex 0
        order = []
        visited = [False] * N
        queue = [0]
        visited[0] = True
        while queue:
            v = queue.pop(0)
            order.append(v)
            for u in ADJ[v]:
                if not visited[u]:
                    visited[u] = True
                    queue.append(u)
        # Append any remaining (shouldn't happen for connected graph)
        for v in range(N):
            if v not in visited:
                order.append(v)

        # MRV-based backtracking with forward checking
        # domains[v] = set of available colors
        domains = [set(range(k)) for _ in range(N)]
        # Initialize: no constraints yet, all colors available

        # For efficiency, track for each vertex which colors are forbidden by neighbors
        # We'll use a recursive backtracking with MRV heuristic

        # Count how many times each vertex is in the order (for tie-breaking)
        order_idx = {v: i for i, v in enumerate(order)}

        def backtrack():
            if time.time() > deadline:
                return None
            # Find unassigned vertex with minimum domain size (MRV)
            min_size = k + 1
            min_v = -1
            for v in order:
                if color[v] == -1:
                    dsize = len(domains[v])
                    if dsize < min_size:
                        min_size = dsize
                        min_v = v
                    if dsize == 0:
                        return None  # dead end
            if min_v == -1:
                # All assigned
                return color[:]

            v = min_v
            # Try each color in domain
            for c in sorted(domains[v]):
                # Check: can we assign c to v without violating adjacency?
                ok = True
                for u in ADJ[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue

                # Tentatively assign
                color[v] = c
                stack_size = len(union_stack)
                valid = True

                # Do unions with colored neighbors
                for u in ADJ[v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        pi = pair_map[(c, cu) if c < cu else (cu, c)]
                        if not do_union(pi, v, u):
                            valid = False
                            break

                if valid:
                    # Forward checking: propagate to unassigned neighbors
                    fc_ok = True
                    saved_domains = {}
                    for u in ADJ[v]:
                        if color[u] == -1:
                            saved_domains[u] = domains[u].copy()
                            domains[u].discard(c)
                            if len(domains[u]) == 0:
                                fc_ok = False
                                break

                    if fc_ok:
                        result = backtrack()
                        if result is not None:
                            return result

                    # Restore domains
                    if fc_ok:
                        for u, old_d in saved_domains.items():
                            domains[u] = old_d

                # Undo unions
                while len(union_stack) > stack_size:
                    undo_union()
                color[v] = -1

            return None

        result = backtrack()
        if result is not None:
            # Final verification
            if not _validate(result):
                raise AssertionError("Validation failed!")
            return [int(x) for x in result]

    # Fallback: use 50 colors (trivially valid)
    return [int(x) for x in range(N)]

# EVOLVE-BLOCK-END
