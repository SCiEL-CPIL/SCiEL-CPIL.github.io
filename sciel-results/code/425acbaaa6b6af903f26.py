"""Rewritten solver: exact branch-and-bound hitting set per graph with deadline guard."""
import time
import random
import sys
sys.setrecursionlimit(100000)

_DEADLINE = 145.0  # safety margin under 150s kill
_start = time.time()

def _elapsed():
    return time.time() - _start

def _deadline_hit():
    return _elapsed() > _DEADLINE

def _build_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _maximal_cliques(adj, n):
    """Bron-Kerbosch with pivot. Yields maximal cliques as frozensets."""
    R = set()
    P = set(range(n))
    X = set()
    results = []

    def bk(R, P, X):
        if not P and not X:
            results.append(frozenset(R))
            return
        # pivot: choose u in P|X maximizing |P ∩ N(u)|
        up = P | X
        if up:
            u = max(up, key=lambda w: len(P & adj[w]))
            candidates = P - adj[u]
        else:
            candidates = P
        for v in list(candidates):
            bk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    bk(R, P, X)
    return results

def _greedy_upper(cliques):
    """Greedy hitting set: repeatedly pick vertex covering most uncovered cliques."""
    uncovered = list(cliques)
    # map vertex -> list of clique indices
    v2c = {}
    for idx, c in enumerate(cliques):
        for v in c:
            v2c.setdefault(v, []).append(idx)
    chosen = []
    # work with a boolean array of covered
    covered = [False] * len(cliques)
    rem = len(cliques)
    # maintain counts
    cnt = {v: len(v2c[v]) for v in v2c}
    # We'll just recompute coverage by scanning uncovered cliques each step (cliques count is small)
    while rem > 0:
        # find vertex covering most uncovered cliques
        best_v = None
        best_cnt = -1
        for v in v2c:
            c = 0
            for idx in v2c[v]:
                if not covered[idx]:
                    c += 1
            if c > best_cnt:
                best_cnt = c
                best_v = v
        if best_v is None or best_cnt == 0:
            break
        chosen.append(best_v)
        for idx in v2c[best_v]:
            if not covered[idx]:
                covered[idx] = True
                rem -= 1
    return chosen

def _disjoint_lower(cliques, covered):
    """Greedy maximal packing of pairwise-disjoint uncovered cliques."""
    # list of uncovered cliques
    unc = [c for i, c in enumerate(cliques) if not covered[i]]
    # sort by size desc
    unc.sort(key=len, reverse=True)
    chosen = []
    used = set()
    for c in unc:
        if c & used:
            continue
        chosen.append(c)
        used |= c
    return len(chosen)

def _solve_hitting(cliques, deadline_check):
    """Exact branch-and-bound minimum hitting set. Returns list of vertices."""
    m = len(cliques)
    if m == 0:
        return []
    # initial upper bound via greedy
    ub = _greedy_upper(cliques)
    best = list(ub)
    best_size = len(best)
    if best_size == 0:
        return []

    # Precompute for each vertex the list of clique indices it covers
    v2c = {}
    for idx, c in enumerate(cliques):
        for v in c:
            v2c.setdefault(v, []).append(idx)

    # Order cliques by size desc for branching priority
    order = sorted(range(m), key=lambda i: (-len(cliques[i]), i))

    covered = [False] * m
    rem = m

    # We'll do DFS. State: covered array (copy-on-write via a set of chosen vertices is easier)
    # Use chosen set; covered derived. For speed, keep covered list and backtrack.
    chosen = []

    # Sort candidate vertices within a clique by descending coverage (heuristic)
    def clique_branch_cliques(idx):
        c = cliques[idx]
        # sort vertices by how many currently-uncovered cliques they cover (desc)
        verts = list(c)
        verts.sort(key=lambda v: sum(1 for ii in v2c[v] if not covered[ii]), reverse=True)
        return verts

    # Precompute a static ordering of vertices per clique by total degree in incidence (desc)
    static_order = {}
    for idx, c in enumerate(cliques):
        verts = list(c)
        verts.sort(key=lambda v: len(v2c[v]), reverse=True)
        static_order[idx] = verts

    # Lower bound function using current covered
    def lower_bound():
        # greedy disjoint packing on uncovered
        unc = [cliques[i] for i in order if not covered[i]]
        # already sorted by size desc because order is size desc
        used = set()
        cnt = 0
        for c in unc:
            if c & used:
                continue
            cnt += 1
            used |= c
            if len(chosen) + cnt >= best_size:
                break
        return cnt

    # DFS with explicit stack to avoid recursion depth issues and allow deadline checks
    # We'll implement recursive with a node counter and deadline guard.
    node_count = 0
    MAX_NODES = 4000000  # cap to stay within time

    # Use recursion with sys.setrecursionlimit already high
    def dfs():
        nonlocal best, best_size, node_count
        if deadline_check() or node_count > MAX_NODES:
            return
        node_count += 1
        # find first uncovered clique in order
        target = -1
        for i in order:
            if not covered[i]:
                target = i
                break
        if target == -1:
            # all covered
            if len(chosen) < best_size:
                best = list(chosen)
                best_size = len(chosen)
            return
        # lower bound prune
        lb = lower_bound()
        if len(chosen) + lb >= best_size:
            return
        # branch on vertices of cliques[target]
        verts = static_order[target]
        for v in verts:
            if v in chosen_set:
                continue
            # include v
            chosen_set.add(v)
            chosen.append(v)
            # mark covered
            newly = []
            for ii in v2c[v]:
                if not covered[ii]:
                    covered[ii] = True
                    newly.append(ii)
            dfs()
            # backtrack
            for ii in newly:
                covered[ii] = False
            chosen.pop()
            chosen_set.remove(v)
            if deadline_check() or node_count > MAX_NODES:
                return

    chosen_set = set()
    dfs()
    return best

def construct():
    global _start
    _start = time.time()
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    for n, p, seed in instances:
        if _deadline_hit():
            # fallback: greedy on this and remaining
            adj = _build_graph(n, p, seed)
            clqs = _maximal_cliques(adj, n)
            clqs = [c for c in clqs if len(c) >= 2]
            # drop redundant
            clqs = [c for i, c in enumerate(clqs) if not any(c < d for j, d in enumerate(clqs) if i != j)]
            res = _greedy_upper(clqs)
            results.append(sorted(res))
            continue
        adj = _build_graph(n, p, seed)
        clqs = _maximal_cliques(adj, n)
        clqs = [c for c in clqs if len(c) >= 2]
        # drop redundant cliques (those contained in another)
        clqs = [c for i, c in enumerate(clqs) if not any(c < d for j, d in enumerate(clqs) if i != j)]
        # remove isolated vertices (not in any clique) - already handled
        res = _solve_hitting(clqs, _deadline_hit)
        results.append(sorted(res))
    return results

# EVOLVE-BLOCK-END
