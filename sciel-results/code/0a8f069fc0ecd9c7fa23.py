"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import random as _random

# EVOLVE-BLOCK-START

_DEADLINE = _time.time() + 108.0          # safety margin vs. the 120 s wall-clock kill
_DEADLINE_DEADLINE = _time.time() + 117.0  # hard internal stop


def _now():
    return _time.time()


def _time_left():
    return _DEADLINE - _now()


# ---------------------------------------------------------------------------
# Exact girth check via non-backtracking walks from every vertex (per contract).
# Returns True iff there is no cycle shorter than `g`.
# ---------------------------------------------------------------------------
def _girth_ok(adj, n, g):
    # non-backtracking walks (u, v) = "arrived at v from u"
    # a walk of length L corresponds to a cycle of length L if it can close.
    # We track the set of non-backtracking walks up to length g-1.
    # A cycle of length c (c < g) exists iff there is a non-backtracking walk
    # of length c-1 from some (u,v) back to a neighbour that closes it.
    # Efficient: BFS over directed edges (states), depth up to g-1, detect return.
    if g <= 3:
        # any edge gives a 2-cycle in a multigraph; simple graph has no 2-cycles.
        # a triangle (3-cycle) check:
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u:
                        return False
        return True
    # states: (u, v) meaning we are at v, came from u.  depth d means path length d.
    # start: for every directed edge (u,v), state (u,v) at depth 1.
    # we can extend (u,v)->(v,w) for w in adj[v], w != u.
    # a cycle of length c exists iff for some state (u,v) at depth c-1, u is in adj[v]
    #   (i.e. we can go v->u closing the loop) and c <= g-1.
    # Actually depth counts edges in the walk; a non-backtracking walk of length c
    # that returns to its start vertex forms a cycle of length c.
    # We detect: at depth L, state (a, b) means walk of length L ending ...->a->b.
    #   It closes a cycle of length L+1 if a in adj[b] (go b->a) ... but a->b->a is
    #   a 2-step backtrack, not a cycle.  Correct closure: the walk started at some
    #   vertex s; we need to track the start.
    # Simpler and exact: enumerate non-backtracking walks up to length g-1 and check
    #   whether the walk's current vertex equals its start vertex.
    # To keep it fast we cap total work; for n<=~400 and g=10 this is fine.
    # We do a BFS keyed by (start, prev, cur) but that's O(n * states).  Instead we
    # do the standard: for each start vertex, BFS non-backtracking, depth < g,
    # and if we reach start again -> short cycle.
    for s in range(n):
        # frontier of (prev, cur) at current depth
        frontier = [(-1, v) for v in adj[s]]  # depth 1
        depth = 1
        seen = set(frontier)
        while frontier and depth < g - 1:
            nxt = []
            for prev, cur in frontier:
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if w == s:
                        return False  # cycle of length depth+1 < g
                    st = (cur, w)
                    if st not in seen:
                        seen.add(st)
                        nxt.append(st)
            frontier = nxt
            depth += 1
    return True


def _build_adj(n, edges, k):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    # verify regular
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return [sorted(a) for a in adj]


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar construction (the seed), kept as a valid fallback.
# ---------------------------------------------------------------------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Randomized 4-regular graph search with local repair for short cycles.
# ---------------------------------------------------------------------------
def _rand_reg(n, k, rng):
    # generate a random k-regular simple graph via configuration model + repair
    # (for small n use a direct approach: random degree sequence matching)
    # Use the standard "random regular graph" via edge swaps from a base.
    # Base: ring graph (2-regular) then add k-2 random edges, then fix.
    # Simpler: build by pairing half-edges with rejection for loops/dups.
    if k % 2 == 1:
        return None
    if n * k < 2 * (k * (k - 1)) // 2:
        return None
    # configuration model
    half = []
    for v in range(n):
        for _ in range(k):
            half.append(v)
    rng.shuffle(half)
    edges = set()
    ok = True
    for i in range(0, n * k, 2):
        a, b = half[i], half[i + 1]
        if a == b:
            ok = False
            break
        if (a, b) in edges or (b, a) in edges:
            ok = False
            break
        edges.add((a, b))
    if not ok:
        return None
    return sorted(edges)


def _short_cycles(adj, n, g):
    """Return list of short cycles (each a list of vertices) of length < g."""
    cycles = []
    # find cycles of length up to g-1 via non-backtracking BFS per start,
    # but only record the first few to bound cost.
    for s in range(n):
        # BFS non-backtracking, depth up to g-1
        # state: (prev, cur, path)
        frontier = [(-1, v, [s, v]) for v in adj[s]]
        depth = 1
        seen = set(frontier[:2])
        while frontier and depth < g - 1:
            nxt = []
            for prev, cur, path in frontier:
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if w == s:
                        cycles.append(path)
                        continue
                    st = (cur, w)
                    if st not in seen:
                        seen.add(st)
                        nxt.append((cur, w, path + [w]))
            frontier = nxt
            depth += 1
            if len(cycles) > 200:
                return cycles
    return cycles


def _repair(adj, n, k, g, rng, max_iters=2000):
    """Try to eliminate all cycles shorter than g via edge swaps."""
    adj = [set(a) for a in adj]
    for it in range(max_iters):
        if _time_left() < 3.0:
            break
        cycles = _short_cycles(adj, n, g)
        if not cycles:
            return adj
        # take the shortest cycle found and try to break it with an edge swap
        cyc = min(cycles, key=len)
        L = len(cyc)
        # pick an edge on the cycle to remove, then reconnect to break the cycle
        # remove edge (cyc[i], cyc[i+1]) and add a random cross edge.
        i = rng.randrange(L)
        a, b = cyc[i], cyc[(i + 1) % L]
        # remove a-b
        adj[a].discard(b)
        adj[b].discard(a)
        # need to add an edge (x,y) with x in adj[a]?? No: we must keep degree k.
        # a lost one neighbour, b lost one. Add a-x and b-y with x,y not adjacent.
        # Simplest: find x in adj[a], y in adj[b] with x!=y, not adjacent to b and a
        # respectively, and x-y not present.  Then swap: remove a-b, add a-y? 
        # That changes degrees.  Proper 2-switch: remove a-b and c-d, add a-c,b-d.
        # Find c,d not on constraints.
        # We'll do a 2-switch: pick c in adj[a], d in adj[b], c!=d, c not adj b, d not adj a,
        # then remove a-b, c-d; add a-c, b-d.
        cand_a = list(adj[a])
        cand_b = list(adj[b])
        done = False
        for _ in range(50):
            c = rng.choice(cand_a)
            d = rng.choice(cand_b)
            if c == d:
                continue
            if d in adj[a]:
                continue
            if c in adj[b]:
                continue
            # perform 2-switch
            adj[a].discard(b); adj[b].discard(a)
            adj[a].discard(c); adj[c].discard(a)
            adj[b].discard(d); adj[d].discard(b)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(d); adj[d].add(b)
            done = True
            break
        if not done:
            # revert
            adj[a].add(b); adj[b].add(a)
    return adj


def _random_search(k, g, target_n, rng, time_budget):
    """Try to find a k-regular girth-g graph with n < target_n.  Return (n, edges) or None."""
    # candidate sizes: try a few below target_n, spaced out
    # Moore bound for (4,10): 1 + 4*sum_{i=0}^{3} 3^i = 1+4*(1+3+9+27)=1+4*40=161
    candidates = []
    for n in range(target_n - 1, max(150, target_n - 60), -1):
        if n % 2 == 0:  # k even, n*k even always; prefer even n
            candidates.append(n)
    if not candidates:
        candidates = [target_n - 2, target_n - 4, target_n - 6]
    rng.shuffle(candidates)
    for n in candidates:
        if _time_left() < 8.0:
            break
        for attempt in range(40):
            if _time_left() < 8.0:
                break
            base = _rand_reg(n, k, rng)
            if base is None:
                continue
            adj = [set() for _ in range(n)]
            for i, j in base:
                adj[i].add(j)
                adj[j].add(i)
            adj = _repair(adj, n, k, g, rng, max_iters=1500)
            # verify
            if all(len(a) == k for a in adj) and _girth_ok(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    return None


def construct(k, g):
    global _DEADLINE, _DEADLINE_DEADLINE
    _DEADLINE = _time.time() + 108.0
    _DEADLINE_DEADLINE = _time.time() + 117.0

    # 1) seed construction (valid baseline)
    try:
        n_seed, edges_seed = _luw(k, g)
    except Exception:
        n_seed, edges_seed = None, None
    seed_valid = False
    if n_seed is not None:
        adj = _build_adj(n_seed, edges_seed, k)
        seed_valid = (adj is not None) and _girth_ok(adj, n_seed, g)
    best_n, best_edges = n_seed, edges_seed
    if not seed_valid:
        # extremely unlikely; keep whatever the seed gave as a last resort
        pass

    # 2) randomized search for something smaller
    rng = _random.Random(12345)
    found = None
    if seed_valid and n_seed is not None:
        try:
            found = _random_search(k, g, n_seed, rng, time_budget=_time_left() - 5.0)
        except Exception:
            found = None
    if found is not None:
        n_f, e_f = found
        if n_f < (best_n if best_n is not None else float('inf')):
            best_n, best_edges = n_f, e_f

    if best_edges is None:
        # last-ditch: return seed even if unverified
        return n_seed, edges_seed
    return best_n, best_edges

# EVOLVE-BLOCK-END
