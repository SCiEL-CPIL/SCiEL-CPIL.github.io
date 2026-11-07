"""Find large clique subdivisions in the pinned graph via clique in G^2."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    # square graph adjacency: distance 1 or 2
    sq = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            sq[u].add(v)
            for w in adj[v]:
                if w != u:
                    sq[u].add(w)
    return adj, sq


def _greedy_clique(sq, n):
    """Greedy maximal clique in sq using degree ordering."""
    order = sorted(range(n), key=lambda x: len(sq[x]), reverse=True)
    C = []
    cand = list(order)
    while cand:
        # pick vertex of max degree within candidate set
        best = None
        bestdeg = -1
        for v in cand:
            d = len(sq[v] & set(cand))
            if d > bestdeg:
                bestdeg = d
                best = v
        if best is None:
            break
        C.append(best)
        cand = [v for v in cand if v in sq[best]]
    return C


def _max_clique_sq(sq, n, deadline):
    """Branch-and-bound maximum clique in sq (Bron-Kerbosch style with coloring)."""
    best = []

    def rec(cand, clique):
        nonlocal best
        if time.time() > deadline:
            return
        if not cand:
            if len(clique) > len(best):
                best = list(clique)
            return
        # if clique + remaining can't beat best, prune
        if len(clique) + len(cand) <= len(best):
            return
        # coloring bound: greedy color cand
        # order cand by degree in cand
        cand_set = set(cand)
        order = sorted(cand, key=lambda x: len(sq[x] & cand_set), reverse=True)
        # greedy coloring
        color = {}
        used_colors = set()
        for v in order:
            neighbors_in_cand = sq[v] & cand_set
            # find smallest color not used by its already-colored neighbors
            c = 0
            while c in used_colors:
                c += 1
            # check against colored neighbors
            # simple: collect colors of colored neighbors
            col_used = set()
            for u, cu in color.items():
                if u in neighbors_in_cand:
                    col_used.add(cu)
            c = 0
            while c in col_used:
                c += 1
            color[v] = c
            used_colors.add(c)
        max_color = max(color.values())
        if len(clique) + max_color <= len(best):
            return
        # process in reverse color order
        for v in reversed(order):
            if v not in cand_set:
                continue
            cand_set.discard(v)
            new_clique = clique + [v]
            new_cand = [u for u in cand_set if u in sq[v]]
            rec(new_cand, new_clique)

    rec(list(range(n)), [])
    return best


def _route_paths(branches, adj, n):
    """Given branch set (pairwise dist<=2), find pairwise-disjoint interior paths.
    Returns list of paths (each a list from branch a to branch b) or None."""
    B = set(branches)
    # We need for every unordered pair (a,b) a simple path whose interior is disjoint
    # from all other interiors and from B.
    # Since dist(a,b)<=2, each path is length 1 (a-b edge) or length 2 (a-x-b).
    # Strategy: for pairs that are edges, use direct edge. For non-edge pairs,
    # we need a common neighbor not in B and not used by another path.
    # Model as matching/assignment: each non-edge pair needs a distinct interior vertex
    # that is a common neighbor of its two endpoints (and not in B).
    pairs = list(combinations(branches, 2))
    # classify
    edge_pairs = []
    need_pairs = []  # (a, b, candidate_interiors)
    for a, b in pairs:
        if b in adj[a]:
            edge_pairs.append((a, b))
        else:
            common = (adj[a] & adj[b]) - B
            need_pairs.append((a, b, common))
    # Now assign distinct interior vertices to need_pairs.
    # This is a bipartite matching: left = need_pairs, right = interior vertices.
    # Use greedy + backtracking (small counts).
    need_pairs = [(a, b, sorted(c)) for a, b, c in need_pairs]

    # Try to find a system of distinct representatives via backtracking
    assign = {}
    used = set()

    def bt(i):
        if i == len(need_pairs):
            return True
        a, b, cands = need_pairs[i]
        # order candidates: prefer vertices with fewer other uses (not needed here, all distinct)
        for x in cands:
            if x in used:
                continue
            used.add(x)
            assign[(a, b)] = x
            if bt(i + 1):
                return True
            used.discard(x)
            del assign[(a, b)]
        return False

    if need_pairs and not bt(0):
        return None

    # Build paths
    paths = []
    for a, b in edge_pairs:
        paths.append([a, b])
    for (a, b), x in assign.items():
        paths.append([a, x, b])
    return paths


def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    adj, sq = _build_graph()
    n = P['n']

    # Start with a good greedy clique
    cur = _greedy_clique(sq, n)
    cur = sorted(cur)

    # Try to improve with branch-and-bound if time permits
    if len(cur) < n and time.time() < deadline - 30:
        mc = _max_clique_sq(sq, n, deadline - 10)
        if len(mc) > len(cur):
            cur = sorted(mc)

    # Now try to extend the current clique greedily if possible
    # (in case max-clique pruned or we want a larger set that still routes)
    # Attempt greedy extension: add vertices that are sq-adjacent to all in cur
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        cur_set = set(cur)
        for v in range(n):
            if v in cur_set:
                continue
            if all(v in sq[u] for u in cur):
                cur.append(v)
                cur_set.add(v)
                improved = True
                break

    # Find the largest routable subset of cur.
    # Try full set first, then drop vertices.
    def try_subset(sub):
        paths = _route_paths(sub, adj, n)
        return paths

    best_paths = None
    cur = sorted(cur)
    # try full
    if len(cur) >= 2:
        best_paths = try_subset(cur)
    if best_paths is None:
        # try dropping one vertex at a time
        for skip in range(len(cur)):
            sub = [v for i, v in enumerate(cur) if i != skip]
            if len(sub) < 2:
                continue
            p = try_subset(sub)
            if p is not None:
                best_paths = p
                break
    if best_paths is None:
        # fallback: find a small routable set via greedy growth
        branches = []
        # start with a vertex, grow
        for start in range(n):
            branches = [start]
            for v in range(n):
                if v in branches:
                    continue
                if all(v in sq[u] for u in branches):
                    branches.append(v)
            if len(branches) >= 2:
                p = try_subset(sorted(branches))
                if p is not None:
                    best_paths = p
                    break

    if best_paths is None:
        # ultimate fallback: an edge
        a, b = P['edges'][0]
        best_paths = [[a, b]]
        branches = [a, b]
    else:
        # recover branches from paths
        branches = set()
        for p in best_paths:
            branches.add(p[0])
            branches.add(p[-1])
        branches = sorted(branches)

    # Validate types
    branches = [int(x) for x in branches]
    paths = [[int(x) for x in p] for p in best_paths]

    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
