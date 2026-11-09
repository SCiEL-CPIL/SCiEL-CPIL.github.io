"""Rewritten program: backtracking with incremental bicolored-cycle detection via union-find."""

# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = [[] for _ in range(N)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

degree = [len(adj[i]) for i in range(N)]
order = sorted(range(N), key=lambda i: -degree[i])


def construct():
    deadline = time.time() + 85.0

    def make_dsu():
        parent = {}
        rank = {}
        stack = []

        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            r = x
            while parent[r] != r:
                r = parent[r]
            return r

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            stack.append((ry, rx, rank[rx]))
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        def undo_one():
            if stack:
                ry, rx, old_rank = stack.pop()
                parent[ry] = ry
                rank[rx] = old_rank

        return find, union, undo_one

    def solve(k):
        color = [-1] * N
        dsus = {}

        def get_dsu(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in dsus:
                dsus[key] = make_dsu()
            return dsus[key]

        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == N:
                return color[:]

            v = order[idx]
            for c in range(k):
                # Check adjacency
                ok = True
                for u in adj[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue

                color[v] = c
                done_unions = []
                feasible = True
                for u in adj[v]:
                    d = color[u]
                    if d == c:
                        continue
                    key = (min(c, d), max(c, d))
                    find, union, undo_one = get_dsu(key)
                    if not union(v, u):
                        feasible = False
                        break
                    done_unions.append(key)

                if not feasible:
                    for key in reversed(done_unions):
                        dsus[key][2]()
                    color[v] = -1
                    continue

                result = backtrack(idx + 1)
                if result is not None:
                    return result

                for key in reversed(done_unions):
                    dsus[key][2]()
                color[v] = -1

            return None

        return backtrack(0)

    for k in range(3, 20):
        if time.time() > deadline:
            break
        res = solve(k)
        if res is not None:
            # Final verification
            if _verify(res):
                return res

    return list(range(N))


def _verify(coloring):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        c1, c2 = coloring[u], coloring[v]
        if c1 == c2:
            return False
        pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))
    for elist in pair_edges.values():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            r = x
            while parent[r] != r:
                r = parent[r]
            return r
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[ry] = rx
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

# EVOLVE-BLOCK-END
