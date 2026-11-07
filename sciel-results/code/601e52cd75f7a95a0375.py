"""Rewritten program: fast bicolored-forest coloring for circulant graph C(50,{1,7})."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    deadline = time.time() + 85

    # Try increasing color counts
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
        # Try periodic patterns first (very fast)
        res = _try_periodic(n, edges, k)
        if res is not None:
            return res
        # Try backtracking
        res = _backtrack(n, adj, k, deadline)
        if res is not None:
            return res

    # Fallback: greedy with larger palette
    res = _greedy_fallback(n, adj)
    if res is not None:
        return res
    return [int(i) for i in range(n)]


def _try_periodic(n, edges, k):
    """Try colorings of form (a*i + b) % k for coprime a, k."""
    for a in range(1, k):
        for b in range(k):
            color = [(a * i + b) % k for i in range(n)]
            if _is_valid(color, n, edges):
                return color
    return None


def _is_valid(color, n, edges):
    """Check adjacency and bicolored-forest constraints."""
    for u, v in edges:
        if color[u] == color[v]:
            return False

    colors_used = set(color)
    clist = list(colors_used)
    for i in range(len(clist)):
        for j in range(i + 1, len(clist)):
            c1, c2 = clist[i], clist[j]
            # Union-find on subgraph induced by colors c1, c2
            parent = list(range(n))
            rank = [0] * n

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True

            for u, v in edges:
                if (color[u] == c1 or color[u] == c2) and (color[v] == c1 or color[v] == c2):
                    if not union(u, v):
                        return False
    return True


def _backtrack(n, adj, k, deadline):
    """Backtracking with full re-verification for each pair."""
    edges = P['edges']
    color = [-1] * n
    order = sorted(range(n), key=lambda v: -len(adj[v]))

    def check_all_pairs():
        colors_used = set(color)
        clist = list(colors_used)
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                c1, c2 = clist[i], clist[j]
                parent = list(range(n))
                rank = [0] * n

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True

                for u, v in edges:
                    if (color[u] == c1 or color[u] == c2) and (color[v] == c1 or color[v] == c2):
                        if not union(u, v):
                            return False
        return True

    def backtrack(idx):
        if time.time() > deadline - 2:
            return None
        if idx == n:
            return list(color)
        v = order[idx]
        for c in range(k):
            valid = True
            for u in adj[v]:
                if color[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            color[v] = c
            if check_all_pairs():
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            color[v] = -1
        return None

    return backtrack(0)


def _greedy_fallback(n, adj):
    color = [-1] * n
    for v in range(n):
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(50):
            if c not in used:
                color[v] = c
                break
    if -1 in color:
        return None
    return [int(c) for c in color]


# EVOLVE-BLOCK-END
