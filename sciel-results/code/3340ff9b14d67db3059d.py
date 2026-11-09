"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# Precompute adjacency as sets of ints.
_ADJ = None
def _adj():
    global _ADJ
    if _ADJ is None:
        _ADJ = [set() for _ in range(P['n'])]
        for a, b in P['edges']:
            _ADJ[a].add(b)
            _ADJ[b].add(a)
    return _ADJ

def _route(b, c, branches, blocked):
    """Find a simple path from b to c using only interior vertices outside branches
    and outside the reserved set 'blocked'. Returns list of vertices or None."""
    if b == c:
        return None
    start = b
    goal = c
    prev = {start: None}
    q = deque([start])
    found = False
    while q:
        u = q.popleft()
        if u == goal:
            found = True
            break
        for v in _ADJ[u]:
            if v in prev:
                continue
            if v in branches:
                continue
            if v in blocked:
                continue
            prev[v] = u
            q.append(v)
    if not found:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path

def _route_all(branches):
    """Try to route all pair-paths with disjoint interiors. Returns dict of paths or None."""
    branches = list(branches)
    bset = set(branches)
    blocked = set()
    paths = {}
    # Order pairs by how "easy" they are (shorter is easier).
    pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
    # Precompute an approximate difficulty = BFS distance ignoring other paths.
    def _dist(a, b):
        if a == b:
            return 0
        seen = {a}
        dq = deque([a])
        d = 0
        while dq:
            d += 1
            for _ in range(len(dq)):
                u = dq.popleft()
                for v in _ADJ[u]:
                    if v in seen:
                        continue
                    if v in bset:
                        continue
                    if v == b:
                        return d
                    seen.add(v)
                    dq.append(v)
        return 10**9
    pairs.sort(key=lambda p: _dist(p[0], p[1]))
    for i, j in pairs:
        p = _route(branches[i], branches[j], bset, blocked)
        if p is None:
            return None
        paths[(i, j)] = p
        for v in p[1:-1]:
            blocked.add(v)
    return paths

def _build_from_clique(clique):
    """Given a clique (list of branch vertices), try to route. If it fails, prune."""
    branches = list(clique)
    paths = _route_all(branches)
    if paths is not None:
        return branches, paths
    # Greedy prune: remove the branch involved in the most failed pairs.
    bset = set(branches)
    # Try removing each branch and see if the rest routes.
    for cand in list(branches):
        trial = [x for x in branches if x != cand]
        if len(trial) < 2:
            continue
        p2 = _route_all(trial)
        if p2 is not None:
            branches = trial
            paths = p2
            break
    if paths is None:
        return branches[:2], None
    return branches, paths

def _try_add(branches, paths, candidate):
    """Try to add candidate to a working (branches, paths). Returns new (branches, paths) or None."""
    newb = branches + [candidate]
    newp = _route_all(newb)
    if newp is not None:
        return newb, newp
    return None

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    adj = _adj()
    n = P['n']

    # Known 23-vertex clique (all but vertex 2) as a strong seed.
    seed_clique = [v for v in range(n) if v != 2]

    # Build initial solution from the seed clique.
    branches, paths = _build_from_clique(seed_clique)
    if paths is None:
        # Fallback: smallest valid = an edge.
        a, b = P['edges'][0]
        branches = [a, b]
        paths = {(0, 1): [a, b]}

    bset = set(branches)

    # Local search: try to add each non-branch vertex.
    improved = True
    while improved and (time.time() - t0) < DEADLINE:
        improved = False
        # Order candidates by degree descending (heuristic: high-degree is more likely routable).
        cands = [v for v in range(n) if v not in bset]
        cands.sort(key=lambda v: len(adj[v]), reverse=True)
        for cand in cands:
            if (time.time() - t0) > DEADLINE:
                break
            res = _try_add(branches, paths, cand)
            if res is not None:
                branches, paths = res
                bset = set(branches)
                improved = True
                break

    # Assemble result.
    br = [int(x) for x in branches]
    pl = [list(paths[(i, j)]) for i in range(len(br)) for j in range(i + 1, len(br))]
    return {'branches': br, 'paths': pl}
# EVOLVE-BLOCK-END
