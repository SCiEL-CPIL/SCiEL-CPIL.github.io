"""
Combined approach: vertex-by-vertex backtracking with Path4 precomputation
for C6-free 4-coloring of complete graphs.

Strategy:
- Build K_n vertex by vertex (0, 1, ..., n-1).
- When adding vertex v, assign colors to edges (v, 0), ..., (v, v-1).
- A C6 is formed iff, for some color c, two neighbors i < d of v in color c
  have a simple path of length 4 between them in the existing color-c graph.
- Precompute Path4[c][i][d] for the current subgraph to make the constraint
  check O(1) per (c, i, d) triple.
- Use backtracking with randomized restarts and a time budget.
"""
import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute all 60 distinct Hamiltonian cycles on 6 labeled vertices
    # for final verification
    _cycles_cache = None

    def _get_cycles():
        nonlocal _cycles_cache
        if _cycles_cache is None:
            cycles = []
            seen = set()
            for perm in permutations(range(1, 6)):
                cycle = (0,) + perm
                edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
                if edges not in seen:
                    seen.add(edges)
                    cycles.append(cycle)
            _cycles_cache = cycles
        return _cycles_cache

    def _verify(colors, n):
        """Verify no color class contains a C6."""
        if n < 6:
            return True
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1

        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)

        cycles = _get_cycles()

        for c in range(4):
            for subset in combinations(range(n), 6):
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return False
        return True

    def _precompute_path4(v, adj_list):
        """
        For the subgraph on vertices 0..v-1, compute Path4[c][i][d]:
        True iff there is a simple path of length 4 (5 edges? No, 4 edges = 5 vertices)
        Wait: path of length 4 means 4 edges, 5 vertices: i-a-b-e-d.
        Actually, to form a C6 when adding edge (v,i) and (v,d) in color c,
        we need a path of length 4 from i to d NOT using v.
        A C6 is v-i-a-b-e-d-v, which is 6 edges.
        The path from i to d is i-a-b-e-d, which is 4 edges.
        So Path4[c][i][d] should be True if there's a simple path of exactly 4 edges from i to d.
        """
        Path4 = [[[False] * v for _ in range(v)] for _ in range(4)]
        for c in range(4):
            for i in range(v):
                neighbors_i = adj_list[c][i]
                if not neighbors_i:
                    continue
                for d in range(v):
                    if i == d:
                        continue
                    found = False
                    # Path: i -> a -> b -> e -> d (4 edges, 5 distinct vertices)
                    for a in neighbors_i:
                        if a == d:
                            continue
                        for b in adj_list[c][a]:
                            if b == i or b == d or b == a:
                                continue
                            for e in adj_list[c][b]:
                                if e == i or e == d or e == a or e == b:
                                    continue
                                if d in adj_list[c][e]:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    Path4[c][i][d] = found
        return Path4

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]

        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1)."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            Path4 = _precompute_path4(v, adj_list)

            # We need to assign colors to (v, 0), (v, 1), ..., (v, v-1)
            # The constraint is: for each color c, if (v, i) and (v, d) are both color c,
            # then Path4[c][i][d] must be False.
            # We backtrack over i = 0, 1, ..., v-1.

            def _backtrack(idx):
                if time.time() > deadline:
                    return False

                if idx == v:
                    return True

                i = idx
                # Try each color
                for c in range(4):
                    # Check constraint: for all d < i where adj[d][v] == c,
                    # Path4[c][i][d] must be False
                    valid = True
                    for d in range(i):
                        if adj[d][v] == c:
                            if Path4[c][i][d]:
                                valid = False
                                break
                    if not valid:
                        continue

                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)

                    if _backtrack(idx + 1):
                        return True

                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)

                return False

            return _backtrack(0)

        # Try to build the coloring
        for v in range(1, n_target):
            if time.time() > deadline:
                return None
            if not _assign_vertex(v):
                return None

        # Extract colors in lexicographic order
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(adj[i][j])
        return colors

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 15:
            break

        found = False
        for seed in range(30):
            if time.time() > deadline - 15:
                break

            colors = _search(n, seed, deadline - 15)
            if colors is not None:
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break

        if found:
            break

    if best_n == 0:
        # Fallback: construct a small valid coloring
        # For n=9, use a simple pattern
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((j - i) % 4)
        # Verify
        if not _verify(colors, n):
            # Try another pattern
            n = 8
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append((j - i) % 4)
            if not _verify(colors, n):
                n = 7
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append((j - i) % 4)
        best_n = n
        best_colors = colors

    return {'n': best_n, 'colors': best_colors}
