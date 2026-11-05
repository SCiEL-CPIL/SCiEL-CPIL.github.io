import time
import random

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - time.time()

def _has_c5(adj, n):
    """Return True if graph on n vertices (adj = list of sets) contains a 5-cycle as subgraph."""
    # A C5 exists iff there is a simple cycle of length 5.
    # Use: for each ordered pair (a,b) with edge, look for path b->...->a of length 4
    # that avoids reusing vertices. Efficient enough for n<=35 with pruning.
    # Simpler: check all 5-cycles via DFS from each start with bounded depth.
    for s in range(n):
        # DFS paths of length 4 starting at s, must end at a neighbor of s, all distinct
        # path: s -> v1 -> v2 -> v3 -> v4, with edge v4-s, and {s,v1,v2,v3,v4} distinct
        stack = [(s, [s])]
        # We want exactly 4 steps then check closure.
        # Represent path as tuple.
        for _ in range(4):
            newstack = []
            for (last, path) in stack:
                for nxt in adj[last]:
                    if nxt in path:
                        continue
                    if len(path) == 4:
                        # path has 5 vertices: s..last (4 edges). Check edge last-s.
                        if s in adj[last] and last != s:
                            return True
                        # do not extend beyond 5 vertices
                        continue
                    newstack.append((nxt, path + [nxt]))
            stack = newstack
    return False

def _max_independent(adj, n):
    """Exact max independent set size for n<=35 via branch and bound (bitmask)."""
    # Use bitmasks.
    mask = [0]*n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= (1<<j)
    full = (1<<n)-1
    best = [0]
    # Bron-Kerbosch style for maximum independent set = maximum clique in complement.
    # Build complement adjacency as bitmask.
    comp = [0]*n
    for i in range(n):
        comp[i] = full ^ (1<<i) ^ mask[i]
    # Max clique via branch and bound.
    best_size = 0
    def bk(R, P):
        nonlocal best_size
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
            return
        # bound: if len(R)+popcount(P) <= best_size, prune
        if len(R) + bin(P).count('1') <= best_size:
            return
        # order candidates
        # pick vertex with max degree in P for speed
        v_list = [i for i in range(n) if (P>>i)&1]
        # simple: iterate
        for i in v_list:
            if (P>>i)&1:
                bk(R+[i], P & comp[i])
                P &= ~(1<<i)
    bk([], full)
    return best_size

def _verify(n, edges):
    adj = [set() for _ in range(n)]
    for u,v in edges:
        if u==v: return False
        if not (0<=u<n and 0<=v<n): return False
        adj[u].add(v); adj[v].add(u)
    if _has_c5(adj, n):
        return False
    if _max_independent(adj, n) >= 9:
        return False
    return True

def _k4_union(n):
    """8 disjoint K4s -> 32 vertices, C5-free, alpha=8."""
    edges = []
    for b in range(n//4):
        base = b*4
        for i in range(4):
            for j in range(i+1,4):
                edges.append([base+i, base+j])
    return n, edges

def _try_random_search(target_n, adj_init, time_budget):
    """Local search: start from a C5-free-ish graph, flip edges to reduce C5 and alpha."""
    # This is a heuristic; keep it lightweight.
    best = None
    rng = random.Random(12345)
    deadline = time.time() + time_budget
    while time.time() < deadline:
        # build random graph
        adj = [set() for _ in range(target_n)]
        for i in range(target_n):
            for j in range(i+1, target_n):
                if rng.random() < 0.5:
                    adj[i].add(j); adj[j].add(i)
        # local improvement: remove edges that create C5 (approx), add edges to reduce alpha
        improved = True
        iters = 0
        while improved and iters < 200 and time.time() < deadline:
            improved = False
            iters += 1
            # try removing random edge if it helps C5
            if _has_c5(adj, target_n):
                # find a C5 edge to remove: just remove a random edge
                cand = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i<j: cand.append((i,j))
                if cand:
                    u,v = cand[rng.randrange(len(cand))]
                    adj[u].discard(v); adj[v].discard(u)
                    improved = True
            else:
                # try adding an edge to reduce alpha
                if _max_independent(adj, target_n) >= 9:
                    # add a random non-edge
                    cand = []
                    for i in range(target_n):
                        for j in range(i+1,target_n):
                            if j not in adj[i]: cand.append((i,j))
                    if cand:
                        u,v = cand[rng.randrange(len(cand))]
                        if not _has_c5_after_add(adj, u, v, target_n):
                            adj[u].add(v); adj[v].add(u)
                            improved = True
        if not _has_c5(adj, target_n) and _max_independent(adj, target_n) <= 8:
            edges = []
            for i in range(target_n):
                for j in adj[i]:
                    if i<j: edges.append([i,j])
            return target_n, edges
    return best

def _has_c5_after_add(adj, u, v, n):
    adj[u].add(v); adj[v].add(u)
    r = _has_c5(adj, n)
    adj[u].discard(v); adj[v].discard(u)
    return r

def construct():
    global _START
    _START = time.time()
    # Guaranteed fallback
    n0, edges0 = _k4_union(32)
    best_n, best_edges = n0, edges0
    # Attempt to beat 32 with n=33,34,35 using a time-bounded search.
    remaining = _remaining()
    # Reserve 20s for verification + safety
    search_budget = max(0.0, remaining - 25.0)
    if search_budget > 5.0:
        for target in (33, 34, 35):
            if _remaining() < 25.0:
                break
            per = search_budget / (35 - target + 1)
            res = _try_random_search(target, None, per)
            if res is not None:
                tn, te = res
                if _verify(tn, te):
                    best_n, best_edges = tn, te
                    break  # got a valid larger graph
    # Final safety: ensure fallback is valid (it is by construction)
    return {'n': best_n, 'edges': best_edges}
