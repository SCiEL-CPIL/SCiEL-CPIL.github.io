import math
import time

_START = time.time()
DEADLINE = _START + 78.0  # safety margin under 90s

_N = 256

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _build_conflicts():
    # conflict[i][j] True if i+j is a perfect square (i<j) or i+i square (i==j)
    # We'll store adjacency as list of lists of neighbors (j>i) for i in 0..N-1
    adj = [[] for _ in range(_N)]
    for i in range(_N):
        for j in range(i, _N):
            s = (i + 1) + (j + 1)
            if _is_square(s):
                adj[i].append(j)
                if j != i:
                    adj[j].append(i)
    return adj

def construct():
    deadline = DEADLINE
    adj = _build_conflicts()
    deg = [len(adj[i]) for i in range(_N)]
    # Sort vertices by degree descending for branching heuristic
    order = sorted(range(_N), key=lambda x: -deg[x])
    pos = {v: i for i, v in enumerate(order)}
    # Reindex so that order[0] is 0, order[1] is 1, ...
    remap = [0] * _N
    for idx, v in enumerate(order):
        remap[v] = idx
    N = _N
    # adjacency in new order (only j>i to save space)
    adj2 = [set() for _ in range(N)]
    for i in range(_N):
        for j in adj[i]:
            if j == i:
                continue
            ii, jj = remap[i], remap[j]
            if ii < jj:
                adj2[ii].add(jj)
            else:
                adj2[jj].add(ii)
    adj2 = [sorted(s) for s in adj2]
    # self-loop flag: vertex i has self-conflict if 2*(i+1) is a square
    self_loop = [False] * N
    for i in range(_N):
        if _is_square(2 * (i + 1)):
            self_loop[remap[i]] = True

    # Precompute for each vertex its neighbors bitset (as set)
    # We'll use sets for candidate set operations.

    # Upper bound: greedy coloring on the induced subgraph of candidates
    # Returns (color_count, color_of)
    def _color_bound(cands):
        if not cands:
            return 0, {}
        color_of = {}
        # order by degree within cands descending
        cset = set(cands)
        for v in sorted(cands, key=lambda x: -deg[x]):
            used = set()
            for w in adj2[v]:
                if w in cset:
                    used.add(color_of.get(w, -1))
            # assign smallest non-negative color not in used
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
        return max(color_of.values()) + 1, color_of

    best = []
    best_size = 0
    # Try a greedy heuristic first to seed best
    def _greedy(cands):
        cands = list(cands)
        res = []
        used = set()
        # pick vertices with smallest degree among remaining
        while cands:
            # find min-degree vertex
            v = min(cands, key=lambda x: (len(adj2[x]) - sum(1 for w in adj2[x] if w in used), x))
            if any(w in used for w in adj2[v]):
                break
            if self_loop[v]:
                # cannot include
                used.add(v)
                cands.remove(v)
                continue
            res.append(v)
            used.add(v)
            # remove neighbors
            for w in adj2[v]:
                if w in cands:
                    cands.remove(w)
        return res

    # Initial greedy on all non-self-loop vertices
    initial = [v for v in range(N) if not self_loop[v]]
    g = _greedy(initial)
    if g:
        best = g[:]
        best_size = len(g)

    # Branch and bound with LP relaxation bound
    def _lp_bound(cands):
        # Simple LP: maximize sum x_i s.t. x_i + x_j <= 1 for edges, x_i <= 1
        # Use a greedy fractional coloring bound instead (faster, often tighter)
        # We'll use the same coloring bound as above
        b, _ = _color_bound(cands)
        return b

    # Branch-and-bound using recursive search on a working set
    # State: (cands: list of available vertices, current: list of chosen)
    # We branch on a vertex: include it (remove it and its neighbors) or exclude it.
    # To keep it fast, we use a simple depth-first with pruning.

    def _dfs(cands, current, depth=0):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if not cands:
            if len(current) > best_size:
                best_size = len(current)
                best = current[:]
            return
        # Upper bound: color bound + current size
        ub = _lp_bound(cands) + len(current)
        if ub <= best_size:
            return
        # Choose branching vertex: max degree in cands
        cset = set(cands)
        v = max(cands, key=lambda x: sum(1 for w in adj2[x] if w in cset))
        # Branch 1: include v (if not self-loop)
        if not self_loop[v]:
            # remove v and its neighbors from cands
            new_cands = [w for w in cands if w != v and w not in adj2[v]]
            _dfs(new_cands, current + [v], depth + 1)
        # Branch 2: exclude v
        new_cands = [w for w in cands if w != v]
        _dfs(new_cands, current, depth + 1)

    # Start DFS with all non-self-loop vertices
    _dfs([v for v in range(N) if not self_loop[v]], [])

    # Map back to original labels
    result = [order[v] + 1 for v in best]
    return result

# Precompute to avoid recomputation overhead (optional)
# The function is designed to be called once; we precompute here for speed.
_ADJ = None

def construct():
    deadline = _START + 78.0
    if _ADJ is None:
        adj = _build_conflicts()
        deg = [len(adj[i]) for i in range(_N)]
        order = sorted(range(_N), key=lambda x: -deg[x])
        remap = [0] * _N
        for idx, v in enumerate(order):
            remap[v] = idx
        N = _N
        adj2 = [set() for _ in range(N)]
        for i in range(_N):
            for j in adj[i]:
                if j == i:
                    continue
                ii, jj = remap[i], remap[j]
                if ii < jj:
                    adj2[ii].add(jj)
                else:
                    adj2[jj].add(ii)
        adj2 = [sorted(s) for s in adj2]
        self_loop = [False] * N
        for i in range(_N):
            if _is_square(2 * (i + 1)):
                self_loop[remap[i]] = True
        _ADJ = (adj2, self_loop, order, deg)
    else:
        adj2, self_loop, order, deg = _ADJ
        N = _N

    def _color_bound(cands):
        if not cands:
            return 0
        cset = set(cands)
        color_of = {}
        for v in sorted(cands, key=lambda x: -deg[x]):
            used = set()
            for w in adj2[v]:
                if w in cset:
                    used.add(color_of.get(w, -1))
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
        return max(color_of.values()) + 1

    best = []
    best_size = 0

    def _greedy(cands):
        cands = list(cands)
        res = []
        used = set()
        while cands:
            v = min(cands, key=lambda x: (len([w for w in adj2[x] if w in used]), x))
            if any(w in used for w in adj2[v]):
                break
            if self_loop[v]:
                used.add(v)
                cands.remove(v)
                continue
            res.append(v)
            used.add(v)
            for w in adj2[v]:
                if w in cands:
                    cands.remove(w)
        return res

    initial = [v for v in range(N) if not self_loop[v]]
    g = _greedy(initial)
    if g:
        best = g[:]
        best_size = len(g)

    def _dfs(cands, current, depth=0):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if not cands:
            if len(current) > best_size:
                best_size = len(current)
                best = current[:]
            return
        cset = set(cands)
        ub = _color_bound(cands) + len(current)
        if ub <= best_size:
            return
        v = max(cands, key=lambda x: sum(1 for w in adj2[x] if w in cset))
        if not self_loop[v]:
            nb = set(adj2[v])
            new_cands = [w for w in cands if w != v and w not in nb]
            _dfs(new_cands, current + [v], depth + 1)
        new_cands = [w for w in cands if w != v]
        _dfs(new_cands, current, depth + 1)

    _dfs(initial, [])

    result = [order[v] + 1 for v in best]
    if not result:
        result = [1]
    return result
