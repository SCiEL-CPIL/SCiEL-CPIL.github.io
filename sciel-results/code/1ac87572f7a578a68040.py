"""Few cycle lengths under degree>=3, girth>=4 on 12 vertices."""
import random
import time

_N = 12
_MIN_DEG = 3
_GIRTH = 4
_DEADLINE = 52.0


def _adj_from_edges(edges):
    adj = [set() for _ in range(_N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _valid(adj):
    for i in range(_N):
        if len(adj[i]) < _MIN_DEG:
            return False
    for i in range(_N):
        for j in adj[i]:
            if i in adj[j] and i < j:
                pass
            for k in adj[i]:
                if k != j and k in adj[j]:
                    return False  # triangle
    return True


def _all_cycle_lengths(adj):
    lengths = set()
    n = _N
    # canonical simple-cycle enumeration
    for start in range(n):
        for end in range(start + 1, n):
            for mid in range(start + 1, end):
                if mid in adj[start] and mid in adj[end]:
                    continue
                # DFS from start to end avoiding mid (and start until end)
                seen = [False] * n
                seen[start] = True
                seen[mid] = True
                stack = [start]
                while stack:
                    x = stack.pop()
                    for nb in adj[x]:
                        if nb == start:
                            continue
                        if seen[nb]:
                            if nb == end:
                                # path start..x-end forms a cycle
                                pass
                            continue
                        if nb == end:
                            # record path length: need count of nodes
                            pass
                        seen[nb] = True
                        stack.append(nb)
    return lengths


def _cycle_lengths_fast(adj):
    """Enumerate all simple cycles via DFS with start<node<end canonical rule."""
    lengths = set()
    n = _N
    adj_list = [sorted(adj[i]) for i in range(n)]
    for start in range(n):
        # all paths from start back to start, only nodes > start in interior
        # DFS
        path = [start]
        on_path = [False] * n
        on_path[start] = True
        stack = [start]
        # iterative DFS tracking
        # We'll do recursive-style with explicit stack of (node, iter_index)
        def dfs():
            pass
        # simpler: recursive
        def rec(x, depth):
            for nb in adj_list[x]:
                if nb == start:
                    if depth + 1 >= _GIRTH:
                        lengths.add(depth + 1)
                    continue
                if nb <= start:
                    continue
                if on_path[nb]:
                    continue
                on_path[nb] = True
                path.append(nb)
                rec(nb, depth + 1)
                path.pop()
                on_path[nb] = False
        rec(start, 1)
    return lengths


def _two_copy_k33():
    """Two K3,3 copies: left {0..5} right {6..11}, each left vertex connects to
    the 3 right vertices in its copy plus cross links to guarantee deg>=3."""
    edges = set()
    # copy A: left 0,1,2 <-> right 6,7,8
    for a in range(3):
        for b in range(3):
            edges.add(tuple(sorted((a, 6 + b))))
    # copy B: left 3,4,5 <-> right 9,10,11
    for a in range(3, 6):
        for b in range(9, 12):
            edges.add(tuple(sorted((a, b))))
    # each left vertex has degree 3 (to its copy's right side).
    # each right vertex has degree 3 (to its copy's left side).
    return [list(e) for e in sorted(edges)]


def _distinct_lengths(edges):
    adj = _adj_from_edges(edges)
    if not _valid(adj):
        return None
    return len(_cycle_lengths_fast(adj))


def construct():
    start_time = time.time()
    best = _two_copy_k33()
    best_score = _distinct_lengths(best)  # expect 2 (4-cycles and 6-cycles)
    if best_score is None:
        best_score = 999

    rng = random.Random(12345)
    while time.time() - start_time < _DEADLINE:
        # random bipartite-ish graph with min degree 3, then local search
        # build a random 3-regular-ish graph on 12 vertices (bipartite to avoid triangles)
        left = list(range(6))
        right = list(range(6, 12))
        # give each left vertex 3 random right neighbors, each right 3 left
        # Use a 3-regular bipartite construction via random permutation
        perm = list(range(6))
        rng.shuffle(perm)
        edges = set()
        for i in range(6):
            for j in range(3):
                r = (perm[i] + j) % 6
                edges.add(tuple(sorted((i, 6 + r))))
        # ensure each right has degree 3: with 3-regular bipartite via cyclic shifts this holds
        cand = [list(e) for e in sorted(edges)]
        sc = _distinct_lengths(cand)
        if sc is not None and sc < best_score:
            best = cand
            best_score = sc
            if best_score == 1:
                break
        # local search: try edge swaps to reduce distinct lengths
        for _ in range(40):
            if time.time() - start_time > _DEADLINE:
                break
            # pick random edge and a random non-edge to swap (keep degrees)
            if not cand:
                break
            e = cand[rng.randrange(len(cand))]
            u, v = e
            # find a vertex pair to swap: remove (u,v), add (u', v') keeping degrees
            # simple: try removing (u,v) and adding (u, v2) where v2 not neighbor, and (u2, v)
            # This is complex; instead try a single edge flip in a 3-regular graph:
            # pick edge (u,v); find u2 in N(v)\{u} and v2 in N(u)\{v}; swap to (u,v2),(u2,v)
            adj = _adj_from_edges(cand)
            nu = [x for x in adj[u] if x != v]
            nv = [x for x in adj[v] if x != u]
            if not nu or not nv:
                break
            u2 = rng.choice(nu)
            v2 = rng.choice(nv)
            if v2 == u2 or (u, v2) in [(a, b) for a, b in cand] or (u2, v) in [(a, b) for a, b in cand]:
                continue
            new_cand = [e2 for e2 in cand if e2 != e]
            new_cand.append([u, v2])
            new_cand.append([u2, v])
            # check no duplicates
            if len(set(tuple(sorted(x)) for x in new_cand)) != len(new_cand):
                continue
            sc2 = _distinct_lengths(new_cand)
            if sc2 is not None and sc2 <= best_score:
                cand = new_cand
                if sc2 < best_score:
                    best = new_cand
                    best_score = sc2
                    if best_score == 1:
                        break

    # final type enforcement
    out = []
    for e in best:
        out.append([int(e[0]), int(e[1])])
    # dedupe & sort, ensure no self loops
    seen = set()
    final = []
    for u, v in out:
        if u == v:
            continue
        t = tuple(sorted((u, v)))
        if t in seen:
            continue
        seen.add(t)
        final.append([t[0], t[1]])
    return final
