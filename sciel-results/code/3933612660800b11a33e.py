"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# Precompute adjacency
_adj = [set() for _ in range(P['n'])]
for _a, _b in P['edges']:
    _adj[_a].add(_b)
    _adj[_b].add(_a)
_es = {tuple(sorted((a, b))) for a, b in P['edges']}


def _bfs_shortest(s, t, banned):
    """Return a shortest path (list of vertices) from s to t avoiding banned,
    or None. s and t are not in banned."""
    if s == t:
        return [s]
    from collections import deque
    parent = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in _adj[u]:
            if v in banned or v in parent:
                continue
            parent[v] = u
            q.append(v)
    if t not in parent:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _find_disjoint_paths(branches):
    """Given branch vertices, try to find internally-disjoint simple paths
    between every unordered pair. Returns dict (pair)->path or None."""
    m = len(branches)
    pairs = list(combinations(range(m), 2))
    # Precompute a shortest path for each pair (banned = all branch vertices).
    banned = set(branches)
    pre = {}
    for i, j in pairs:
        p = _bfs_shortest(branches[i], branches[j], banned)
        if p is None:
            return None
        pre[(i, j)] = p
    # Order pairs: those with fewest interior options first (most constrained).
    order = sorted(pairs, key=lambda p: len(pre[p]) - 2)
    # Greedy with backtracking.
    used = set()  # interior vertices already used
    chosen = {}

    def bt(idx):
        if idx == len(order):
            return True
        i, j = order[idx]
        # try a few candidate paths: the precomputed shortest, plus a couple of
        # alternative shortest paths found via BFS with the current used-set.
        cand = []
        p0 = pre[(i, j)]
        cand.append(p0)
        # A few alternatives: BFS avoiding used as well.
        alt = _bfs_shortest(branches[i], branches[j], banned | used)
        if alt is not None:
            cand.append(alt)
        for p in cand:
            interior = set(p[1:-1])
            if interior & used:
                continue
            used.update(interior)
            chosen[(i, j)] = p
            if bt(idx + 1):
                return True
            used.difference_update(interior)
            del chosen[(i, j)]
        return False

    if bt(0):
        return chosen
    return None


def _try_clique(clique):
    paths = _find_disjoint_paths(clique)
    if paths is None:
        return None
    # Build output paths keyed by vertex pairs.
    out_paths = {}
    for (i, j), p in paths.items():
        out_paths[(clique[i], clique[j])] = list(p)
    return list(clique), out_paths


def _max_clique_search(deadline):
    """Bron-Kerbosch with pivot, collect maximal cliques; keep the largest."""
    best = None
    best_size = 0

    # Work with a bitset of all vertices for efficiency.
    n = P['n']
    allv = set(range(n))

    def bk(R, Pset, Xset):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if not Pset and not Xset:
            # maximal clique R
            if len(R) > best_size:
                best_size = len(R)
                best = list(R)
            return
        # pivot
        U = Pset | Xset
        pivot = max(U, key=lambda v: len(_adj[v] & Pset)) if U else None
        candidates = [v for v in Pset if (pivot is None or v not in _adj[pivot])]
        for v in candidates:
            if time.time() > deadline:
                return
            newR = R | {v}
            newP = Pset & _adj[v]
            newX = Xset & _adj[v]
            bk(newR, newP, newX)
            Pset.discard(v)
            Xset.add(v)

    bk(set(), allv, set())
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    # 1) Find the maximum clique.
    maxc = _max_clique_search(deadline)
    if maxc is None or len(maxc) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # 2) Try to realize the max clique as a subdivision; if it fails, drop
    #    vertices greedily and re-try.
    cliques_to_try = []
    cliques_to_try.append(list(maxc))
    # Build descending-size candidate cliques from subsets of the max clique.
    mc = list(maxc)
    for size in range(len(mc) - 1, 1, -1):
        if time.time() > deadline:
            break
        for sub in combinations(mc, size):
            cliques_to_try.append(list(sub))

    best_result = None
    best_k = 0
    for clique in cliques_to_try:
        if time.time() > deadline:
            break
        if len(clique) <= best_k:
            continue
        res = _try_clique(clique)
        if res is not None:
            branches, out_paths = res
            best_result = (branches, out_paths)
            best_k = len(branches)
            if len(branches) == len(mc):
                break  # achieved the max clique size

    if best_result is None:
        # Fall back: find any edge that works (always a 2-branch subdivision).
        u, v = P['edges'][0]
        best_result = ([u, v], {(u, v): [u, v]})

    branches, out_paths = best_result
    branches = [int(b) for b in branches]
    # Order paths in a deterministic order (by sorted branch pair).
    paths_list = []
    for (a, b) in sorted(out_paths.keys()):
        paths_list.append([int(x) for x in out_paths[(a, b)]])
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
