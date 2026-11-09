"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Uses structured greedy + backtracking with fast bicolored-forest verification."""
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)


def _bfs_2color():
    """Return proper 2-coloring if bipartite, else None."""
    color = [-1] * _N
    for start in range(_N):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            cu = color[u]
            for v in _ADJ[u]:
                if color[v] == -1:
                    color[v] = 1 - cu
                    q.append(v)
                elif color[v] == cu:
                    return None
    return color


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


def _greedy_color(k, order=None):
    """Greedy coloring with k colors."""
    if order is None:
        order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
    coloring = [-1] * _N
    for v in order:
        used = set()
        for nb in _ADJ[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
        if coloring[v] == -1:
            return None
    return coloring


def _check_pair_forest(color, ci, cj):
    """Check if subgraph induced by colors ci and cj is a forest."""
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
        ca, cb = color[a], color[b]
        if ca == -1 or cb == -1:
            continue
        if (ca == ci and cb == cj) or (ca == cj and cb == ci):
            if not union(a, b):
                return False
    return True


def _backtrack_k(k, deadline):
    """Backtracking search for k-coloring with bicolored-forest check."""
    color = [-1] * _N
    order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == _N:
            return True
        v = order[pos]
        used = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                used.add(color[nb])
        
        # Sort colors by how many bicolored edges they'd create (fewer is better)
        color_options = [c for c in range(k) if c not in used]
        def color_score(c):
            count = 0
            for nb in _ADJ[v]:
                if color[nb] != -1 and color[nb] != c:
                    count += 1
            return count
        color_options.sort(key=color_score)
        
        for c in color_options:
            color[v] = c
            # Check bicolored forest constraint for all pairs involving c
            ok = True
            for u in _ADJ[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                ci, cj = min(c, cu), max(c, cu)
                if not _check_pair_forest(color, ci, cj):
                    ok = False
                    break
            if ok:
                if backtrack(pos + 1):
                    return True
            color[v] = -1
        return False
    
    if backtrack(0):
        return color[:]
    return None


def _try_k_colors(k, deadline, max_greedy_attempts=20):
    """Try to find valid k-coloring using greedy with multiple orderings, then backtracking."""
    orderings = []
    orderings.append(sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v)))
    orderings.append(sorted(range(_N), key=lambda v: (len(_ADJ[v]), v)))
    orderings.append(list(range(_N)))
    orderings.append(list(range(_N - 1, -1, -1)))
    orderings.append(sorted(range(_N), key=lambda v: (abs(v - 24.5), v)))
    
    for order in orderings:
        if time.time() > deadline:
            break
        c = _greedy_color(k, order)
        if c is not None and _is_valid(c):
            return c
    
    rng = random.Random(12345 + k)
    for attempt in range(max_greedy_attempts):
        if time.time() > deadline:
            break
        order = list(range(_N))
        rng.shuffle(order)
        c = _greedy_color(k, order)
        if c is not None and _is_valid(c):
            return c
    
    if k <= 6:
        result = _backtrack_k(k, deadline)
        if result is not None:
            return result
    
    return None


def construct():
    deadline = time.time() + 85.0
    
    # 1) Check bipartite (2 colors)
    c2 = _bfs_2color()
    if c2 is not None and _is_valid(c2):
        return [int(x) for x in c2]
    
    # 2) Try k=3,4,5,... with greedy first, then backtracking
    best = None
    for k in range(3, 12):
        if time.time() > deadline - 10.0:
            break
        remaining = deadline - 10.0 - time.time()
        if remaining < 2.0:
            break
        found = _try_k_colors(k, time.time() + remaining)
        if found is not None:
            best = found
            break
    
    # 3) Fallback: greedy with more colors
    if best is None:
        for k in range(5, 15):
            if time.time() > deadline - 5.0:
                break
            c = _greedy_color(k)
            if c is not None and _is_valid(c):
                best = c
                break
    
    # 4) Last resort: unique colors
    if best is None:
        best = list(range(_N))
    
    return [int(x) for x in best]
