"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Exploits circulant graph structure C(50, {1,7}) with pattern search and targeted backtracking."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)


def _is_valid(coloring):
    """Check proper coloring and bicolored-forest constraint."""
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    present = sorted(set(coloring))
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ci, cj = present[i], present[j]
            parent = list(range(_N))
            rank = [0] * _N

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

            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if ca in (ci, cj) and cb in (ci, cj):
                    if not union(a, b):
                        return False
    return True


def _backtrack_k(k, deadline):
    """Backtracking search for k-coloring with bicolored-forest constraint."""
    color = [-1] * _N
    order = list(range(_N))

    def check_bicolored_cycle(v, c):
        """Check if assigning color c to v creates a bicolored cycle."""
        for u in _ADJ[v]:
            cu = color[u]
            if cu == -1 or cu == c:
                continue
            # Check if v and u are connected in subgraph of colors {c,cu} without edge (v,u)
            visited = {v}
            stack = [v]
            while stack:
                node = stack.pop()
                for w in _ADJ[node]:
                    if node == v and w == u:
                        continue
                    if node == u and w == v:
                        continue
                    cw = color[w]
                    if cw == -1 or cw not in (c, cu):
                        continue
                    if w == u:
                        return True
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
        return False

    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == _N:
            return True

        v = order[pos]
        forbidden = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                forbidden.add(color[nb])

        for c in range(k):
            if c in forbidden:
                continue
            if check_bicolored_cycle(v, c):
                continue
            color[v] = c
            if backtrack(pos + 1):
                return True
            color[v] = -1
        return False

    if backtrack(0):
        return color[:]
    return None


def construct():
    deadline = time.time() + 85.0

    # Try linear periodic patterns for small k
    for k in range(2, 7):
        for a in range(1, k):
            for b in range(k):
                coloring = [(i * a + b) % k for i in range(_N)]
                if _is_valid(coloring):
                    return [int(x) for x in coloring]

    # Try backtracking for k=3,4,5
    for k in range(3, 6):
        if time.time() > deadline - 5.0:
            break
        result = _backtrack_k(k, deadline - 2.0)
        if result is not None:
            if _is_valid(result):
                return [int(x) for x in result]

    # Fallback: greedy with increasing k
    for k in range(4, 12):
        if time.time() > deadline - 2.0:
            break
        coloring = [-1] * _N
        for v in range(_N):
            used = set()
            for nb in _ADJ[v]:
                if coloring[nb] != -1:
                    used.add(coloring[nb])
            for c in range(k):
                if c not in used:
                    coloring[v] = c
                    break
        if -1 not in coloring and _is_valid(coloring):
            return [int(x) for x in coloring]

    # Last resort
    return [int(x) for x in range(_N)]
