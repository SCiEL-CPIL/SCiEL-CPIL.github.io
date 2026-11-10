"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

DEADLINE = 78.0  # safety margin under 90s

def _now():
    return time.time()

def _bfs_path(s, t, forbidden):
    """Return a simple path s..t avoiding `forbidden` (a set of vertices), or None."""
    if s == t:
        return [s]
    from collections import deque
    prev = {s: None}
    q = deque([s])
    while q:
        x = q.popleft()
        if x == t:
            break
        for y in ADJ[x]:
            if y in prev or y in forbidden:
                continue
            prev[y] = x
            q.append(y)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def _route(B):
    """Try to assign pairwise interior-disjoint paths for all pairs in B.
    Returns dict {(i,j): path} or None. Uses DFS backtracking over pairs,
    reserving interior vertices."""
    k = len(B)
    pairs = list(combinations(range(k), 2))
    # Order pairs to reduce branching: those with fewer options first is complex;
    # use a fixed order but try to reserve.
    reserved = set()
    assignment = {}

    def dfs(idx):
        if _now() > DEADLINE:
            return False
        if idx == len(pairs):
            return True
        i, j = pairs[idx]
        s, t = B[i], B[j]
        # forbidden = reserved interiors (branches other than s,t are always forbidden as interiors)
        forbidden = set(reserved)
        # find all simple paths up to a small length limit to keep it bounded
        # We'll enumerate a few candidate paths via BFS with a length cap.
        cand = _candidate_paths(s, t, forbidden)
        for p in cand:
            interior = set(p[1:-1])
            if interior & reserved:
                continue
            # commit
            reserved.update(interior)
            assignment[(i, j)] = p
            if dfs(idx + 1):
                return True
            reserved -= interior
            del assignment[(i, j)]
        return False

    ok = dfs(0)
    if not ok:
        return None
    return assignment

def _candidate_paths(s, t, forbidden, cap=6):
    """Enumerate simple s-t paths of length <= cap avoiding forbidden, limited count."""
    from collections import deque
    results = []
    # DFS enumeration with depth cap and a result cap
    MAXRES = 40
    def rec(cur, path, used):
        if len(results) >= MAXRES:
            return
        if len(path) - 1 > cap:
            return
        if cur == t:
            results.append(list(path))
            return
        for nb in ADJ[cur]:
            if nb in used or nb in forbidden:
                continue
            used.add(nb)
            path.append(nb)
            rec(nb, path, used)
            path.pop()
            used.discard(nb)
    used = {s}
    rec(s, [s], used)
    # sort by length then lexicographic for determinism
    results.sort(key=lambda p: (len(p), p))
    return results

def _feasible(B):
    return _route(B) is not None

def _grow(B):
    """Greedily add vertices to B while feasible. Returns (B, assignment)."""
    B = list(B)
    for cand in range(N):
        if cand in B:
            continue
        newB = B + [cand]
        a = _route(newB)
        if a is not None:
            B = newB
    return B, _route(B)

def _local_search(B):
    """Try to add a vertex via swap or direct add."""
    B = list(B)
    improved = True
    while improved:
        improved = False
        # try direct add of any non-branch
        for cand in range(N):
            if cand in B:
                continue
            if _route(B + [cand]) is not None:
                B.append(cand)
                improved = True
                break
    return B

def construct():
    t0 = _now()
    best = None
    best_size = 0

    # Seed with cliques (Bron-Kerbosch) for good starting branch sets
    cliques = _cliques()
    # Sort cliques by size desc
    cliques.sort(key=len, reverse=True)

    for c in cliques:
        if _now() > DEADLINE:
            break
        if len(c) + 1 <= best_size:
            continue
        B, a = _grow(c)
        if _now() > DEADLINE:
            break
        B = _local_search(B)
        if len(B) > best_size:
            best = B
            best_size = len(B)
            a = _route(B)
            if a is None:
                # should not happen
                continue
        if best_size >= N:
            break

    # Also try a pure greedy from scratch by degree
    if _now() < DEADLINE:
        order = sorted(range(N), key=lambda v: -len(ADJ[v]))
        B = []
        for cand in order:
            if _now() > DEADLINE:
                break
            if _route(B + [cand]) is not None:
                B.append(cand)
        B = _local_search(B)
        if len(B) > best_size:
            best = B
            best_size = len(B)

    if best is None or len(best) < 2:
        u, v = P['edges'][0]
        best = [u, v]

    a = _route(best)
    if a is None:
        # fallback: take a valid small set
        for c in cliques:
            if len(c) >= 2 and _route(c) is not None:
                best = c
                a = _route(best)
                break
        if a is None:
            u, v = P['edges'][0]
            best = [u, v]
            a = {(0, 1): [u, v]}

    # Build output paths keyed by unordered branch pair
    paths = {}
    for (i, j), p in a.items():
        paths[(i, j)] = p

    # Ensure at least 2 branches
    branches = [int(x) for x in best]
    out_paths = []
    for (i, j) in combinations(range(len(branches)), 2):
        out_paths.append([int(x) for x in paths[(i, j)]])

    return {'branches': branches, 'paths': out_paths}

def _cliques():
    """Bron-Kerbosch to find all maximal cliques."""
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    result = []
    def bk(R, Pset, Xset):
        if _now() > DEADLINE:
            return
        if not Pset and not Xset:
            result.append(list(R))
            return
        # pivot
        P_X = Pset | Xset
        if not P_X:
            piv = None
        else:
            piv = max(P_X, key=lambda v: len(adj[v] & Pset))
            candidates = Pset - adj[piv]
        for v in list(candidates):
            bk(R | {v}, Pset & adj[v], Xset & adj[v])
            Pset.remove(v)
            Xset.add(v)
    bk(set(), set(range(N)), set())
    return result

# EVOLVE-BLOCK-END
