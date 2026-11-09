"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Incremental bicolored-cycle detector with undo support.
    # For each unordered color pair (c1, c2) we maintain:
    #   parent[c1][c2] : union-find parent array (size n)
    #   rank[c1][c2]   : union-find rank array
    #   edge_stack[c1][c2] : list of (u, v) edges added, for undo
    # When we assign a color to a vertex, we add the incident bicolored edges
    # to the appropriate DSUs. If any union finds a cycle, we undo and backtrack.

    class ColorPairState:
        """Manages incremental DSU state for all color pairs up to max_colors."""
        def __init__(self, n, max_colors):
            self.n = n
            self.max_colors = max_colors
            # parent[c1][c2] for c1 < c2
            self.parent = {}
            self.rank = {}
            self.edge_stack = {}
            for c1 in range(max_colors):
                for c2 in range(c1 + 1, max_colors):
                    self.parent[(c1, c2)] = list(range(n))
                    self.rank[(c1, c2)] = [0] * n
                    self.edge_stack[(c1, c2)] = []

        def _find(self, pair, x):
            parent = self.parent[pair]
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(self, pair, x, y):
            """Union x and y in the given pair's DSU. Returns False if cycle."""
            parent = self.parent[pair]
            rank = self.rank[pair]
            px = self._find(pair, x)
            py = self._find(pair, y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        def add_edge(self, u, v, cu, cv):
            """Add edge (u,v) where coloring[u]=cu, coloring[v]=cv. Returns False if cycle."""
            if cu == cv:
                return True  # same-color edge, no bicolored constraint
            c1, c2 = (cu, cv) if cu < cv else (cv, cu)
            pair = (c1, c2)
            if not self._union(pair, u, v):
                return False
            self.edge_stack[pair].append((u, v))
            return True

        def undo_edge(self, u, v, cu, cv):
            """Remove the last-added edge for pair (cu,cv)."""
            if cu == cv:
                return
            c1, c2 = (cu, cv) if cu < cv else (cv, cu)
            pair = (c1, c2)
            self.edge_stack[pair].pop()
            # Rebuild DSU for this pair from remaining edges
            self.parent[pair] = list(range(self.n))
            self.rank[pair] = [0] * self.n
            for eu, ev in self.edge_stack[pair]:
                self._union(pair, eu, ev)

    def check_validity(coloring):
        """Full validation of a complete coloring."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check bicolored subgraphs are forests
        seen_pairs = set()
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    seen_pairs.add((c1, c2))
                else:
                    seen_pairs.add((c2, c1))
        for c1, c2 in seen_pairs:
            parent = list(range(n))
            rank = [0] * n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    pu, pv = find(u), find(v)
                    if pu == pv:
                        return False
                    if rank[pu] < rank[pv]:
                        pu, pv = pv, pu
                    parent[pv] = pu
                    if rank[pu] == rank[pv]:
                        rank[pu] += 1
        return True

    def backtrack_mrv(max_colors, seed, order=None):
        """Backtracking with MRV heuristic and incremental cycle detection."""
        import random
        rng = random.Random(seed)

        if order is None:
            order = list(range(n))
        else:
            order = list(order)

        coloring = [-1] * n
        state = ColorPairState(n, max_colors)
        uncolored = set(range(n))

        def dfs():
            if time.time() > deadline:
                return False
            if not uncolored:
                return True

            # MRV: pick vertex with fewest available colors
            best_v = None
            best_available = None
            best_count = max_colors + 1

            for v in uncolored:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = v
                    best_available = available
                    if best_count == 0:
                        break

            if best_count == 0:
                return False

            # Shuffle available colors for diversity
            rng.shuffle(best_available)

            for c in best_available:
                # Try assigning color c to best_v
                # Check all incident edges to already-colored neighbors
                added_edges = []
                ok = True
                for u in adj[best_v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        if not state.add_edge(best_v, u, c, d):
                            ok = False
                            break
                        added_edges.append((best_v, u, c, d))

                if not ok:
                    # Undo any edges we added before detecting cycle
                    for (eu, ev, ecu, ecv) in added_edges:
                        state.undo_edge(eu, ev, ecu, ecv)
                    continue

                # Assign
                coloring[best_v] = c
                uncolored.discard(best_v)

                if dfs():
                    return True

                # Backtrack
                uncolored.add(best_v)
                coloring[best_v] = -1
                for (eu, ev, ecu, ecv) in reversed(added_edges):
                    state.undo_edge(eu, ev, ecu, ecv)

            return False

        if dfs():
            return coloring[:]
        return None

    def local_repair(coloring, max_colors, max_iters=200):
        """Try to reduce distinct colors by recoloring vertices."""
        if coloring is None:
            return None
        coloring = coloring[:]
        used_colors = set(coloring)

        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for v in range(n):
                old_c = coloring[v]
                # Try to recolor v to a different color that is already used
                # or to a lower color, to potentially reduce distinct count
                neighbor_colors = set(coloring[u] for u in adj[v])
                # Candidate colors: all used colors except old_c, plus maybe lower unused
                candidates = []
                for c in sorted(used_colors):
                    if c != old_c and c not in neighbor_colors:
                        candidates.append(c)
                # Also try colors 0..max_colors-1 not in neighbors and not old_c
                for c in range(max_colors):
                    if c != old_c and c not in neighbor_colors and c not in candidates:
                        candidates.append(c)

                for c in candidates:
                    # Check if recoloring v to c maintains bicolored forest property
                    # We need to verify: for each neighbor u with color d,
                    # the edge (v,u) in pair (c,d) doesn't create a cycle
                    # Simplest: tentatively recolor and do full check
                    coloring[v] = c
                    if check_validity(coloring):
                        # Check if this reduces distinct colors
                        new_used = set(coloring)
                        if len(new_used) < len(used_colors):
                            used_colors = new_used
                            improved = True
                            break
                        # Even if same count, accept if it uses a lower max color
                        elif max(new_used) < max(used_colors):
                            used_colors = new_used
                            improved = True
                            break
                    coloring[v] = old_c
            if not improved:
                break

        return coloring

    best_coloring = None
    best_count = n

    # Strategy 1: Quick greedy attempts with few colors
    for max_colors in [3, 4]:
        if time.time() > deadline:
            break
        coloring = [-1] * n
        for v in range(n):
            neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
            available = [c for c in range(max_colors) if c not in neighbor_colors]
            if not available:
                coloring = None
                break
            coloring[v] = available[0]
        if coloring is not None and check_validity(coloring):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]

    # Strategy 2: Backtracking with MRV — the main workhorse
    # Try 3 colors first with many seeds
    for seed in range(500):
        if time.time() > deadline:
            break
        coloring = backtrack_mrv(3, seed)
        if coloring is not None:
            if not check_validity(coloring):
                continue
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
                # Try local repair to possibly reduce further
                repaired = local_repair(coloring, 3)
                if repaired is not None:
                    rep_used = len(set(repaired))
                    if rep_used < best_count:
                        best_count = rep_used
                        best_coloring = repaired[:]
            if best_count <= 3:
                break

    # If we didn't find 3, try 4 colors
    if best_count > 3:
        for seed in range(500):
            if time.time() > deadline:
                break
            coloring = backtrack_mrv(4, seed)
            if coloring is not None:
                if not check_validity(coloring):
                    continue
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                    repaired = local_repair(coloring, 4)
                    if repaired is not None:
                        rep_used = len(set(repaired))
                        if rep_used < best_count:
                            best_count = rep_used
                            best_coloring = repaired[:]
                if best_count <= 4:
                    break

    # Strategy 3: Try different vertex orderings with backtracking
    if best_count > 3:
        import random
        for seed in range(200):
            if time.time() > deadline:
                break
            order = list(range(n))
            random.seed(seed * 7919 + 13)
            random.shuffle(order)
            coloring = backtrack_mrv(3, seed, order)
            if coloring is not None:
                if not check_validity(coloring):
                    continue
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= 3:
                    break

    # Fallback: greedy with more colors
    if best_coloring is None:
        for max_colors in [5, 6, 7, 8]:
            coloring = [-1] * n
            for v in range(n):
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if not available:
                    coloring = None
                    break
                coloring[v] = available[0]
            if coloring is not None and check_validity(coloring):
                best_coloring = coloring
                best_count = len(set(coloring))
                break

    if best_coloring is None:
        best_coloring = list(range(n))
        best_count = n

    # Convert to proper Python ints
    result = [int(c) for c in best_coloring]

    # Verify one final time
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49

    # Final validity check
    for u, v in edges:
        assert result[u] != result[v], f"Adjacent vertices {u},{v} have same color {result[u]}"

    # Check bicolored forests
    seen_pairs = set()
    for u, v in edges:
        c1, c2 = result[u], result[v]
        if c1 != c2:
            if c1 < c2:
                seen_pairs.add((c1, c2))
            else:
                seen_pairs.add((c2, c1))
    for c1, c2 in seen_pairs:
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges:
            cu, cv = result[u], result[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                pu, pv = find(u), find(v)
                assert pu != pv, f"Bicolored cycle in colors {c1},{c2}"
                if rank[pu] < rank[pv]:
                    pu, pv = pv, pu
                parent[pv] = pu
                if rank[pu] == rank[pv]:
                    rank[pu] += 1

    return result
# EVOLVE-BLOCK-END
