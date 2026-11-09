"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    t0 = time.time()
    TIME_LIMIT = 58.0

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Pre-compute edge list for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    # Union-Find with undo for each color pair
    class UF:
        __slots__ = ('parent', 'rank', 'journal')
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.journal = []

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.journal.append((ry, self.parent[ry], self.rank[ry], rx, self.parent[rx], self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True

        def undo(self):
            if not self.journal:
                return
            ry, pr, rr, rx, prx, rrx = self.journal.pop()
            self.parent[ry] = pr
            self.rank[ry] = rr
            self.parent[rx] = prx
            self.rank[rx] = rrx

    # Pre-allocate UFs for all possible pairs
    uf_pool = {}
    def get_uf(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in uf_pool:
            uf_pool[key] = UF(n)
        return uf_pool[key]

    def solve(k, seed):
        rng = random.Random(seed)
        color = [-1] * n
        # Symmetry breaking: fix vertex 0 to color 0
        color[0] = 0
        # For k >= 2, fix vertex 1 (adjacent to 0) to color 1
        if k >= 2:
            color[1] = 1

        # Track which UF unions were made for each assigned vertex
        # For each vertex v, we need to track which (uf_key, uf) pairs were unioned
        vertex_unions = [[] for _ in range(n)]

        # For MRV, we need to quickly compute available colors
        # We'll maintain a count of how many times each color appears in neighbors
        # But the bicolored constraint makes this complex, so we just compute on the fly

        def is_valid_assignment(v, c):
            # Check all neighbors
            for u in adj[v]:
                cu = color[u]
                if cu == -1:
                    continue
                if cu == c:
                    return False
                key = (min(cu, c), max(cu, c))
                uf = get_uf(cu, c)
                if uf.find(u) == uf.find(v):
                    return False
            return True

        def assign(v, c):
            color[v] = c
            vertex_unions[v] = []
            for u in adj[v]:
                cu = color[u]
                if cu == -1:
                    continue
                key = (min(cu, c), max(cu, c))
                uf = get_uf(cu, c)
                if not uf.union(u, v):
                    # Should not happen if is_valid_assignment was called
                    return False
                vertex_unions[v].append((key, uf))
            return True

        def unassign(v):
            for key, uf in reversed(vertex_unions[v]):
                uf.undo()
            vertex_unions[v] = []
            color[v] = -1

        def count_available(v):
            count = 0
            for c in range(k):
                if is_valid_assignment(v, c):
                    count += 1
            return count

        def backtrack(assigned):
            if time.time() - t0 > TIME_LIMIT:
                return -1  # timeout
            if assigned == n:
                return 0

            # MRV: find unassigned vertex with fewest available colors
            best_v = -1
            best_count = k + 1
            best_colors = []

            for v in range(n):
                if color[v] != -1:
                    continue
                avail = []
                for c in range(k):
                    if is_valid_assignment(v, c):
                        avail.append(c)
                if len(avail) < best_count:
                    best_count = len(avail)
                    best_v = v
                    best_colors = avail
                    if best_count == 0:
                        break

            if best_v == -1:
                return 0  # all assigned
            if best_count == 0:
                return 1  # dead end

            rng.shuffle(best_colors)
            for c in best_colors:
                if not assign(best_v, c):
                    continue
                res = backtrack(assigned + 1)
                if res == 0:
                    return 0
                if res == -1:
                    unassign(best_v)
                    return -1
                unassign(best_v)
            return 1

        res = backtrack(2 if k >= 2 else 1)
        if res == 0:
            return color
        return None

    # Try increasing number of colors
    for k in range(3, 13):
        for seed in range(30):
            if time.time() - t0 > TIME_LIMIT:
                break
            result = solve(k, seed)
            if result is not None:
                # Verify
                valid = True
                for u, v in edges:
                    if result[u] == result[v]:
                        valid = False
                        break
                if valid:
                    uf_data = {}
                    for u, v in edges:
                        cu, cv = result[u], result[v]
                        if cu == cv:
                            continue
                        key = (min(cu, cv), max(cu, cv))
                        if key not in uf_data:
                            uf_data[key] = UF(n)
                        uf = uf_data[key]
                        if not uf.union(u, v):
                            valid = False
                            break
                    if valid:
                        return [int(c) for c in result]
        if time.time() - t0 > TIME_LIMIT:
            break

    # Fallback: 50 colors (identity mapping)
    return list(range(n))

# EVOLVE-BLOCK-END
