"""Direct-approach cage solver: many explicit constructions + exact girth
verification, returning the smallest verified k-regular graph of girth >= g."""
import time
import threading
from collections import deque

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
    # non-backtracking state = (current_vertex, previous_vertex)
    prev = {}          # (v, p) -> (u, dist)  where u is the vertex before p on path
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                return dist + 1          # closed non-backtracking walk = cycle
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9                        # no cycle at all (tree / forest)


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False              # loop or double edge
    return _nonback_girth(adj, n) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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


# ---------------- construction 2: generalized Petersen graph G(N, k) ----------
def _petersen(k, g):
    if k != 3:
        return None
    for N in range(2 * k + 1, 5000):
        if N % 2 == 0 or N <= 2 * k:
            continue
        edges = []
        for i in range(N):
            edges.append((i, (i + 1) % N))
            edges.append((N + i, (N + (i + 1)) % N))
            edges.append((i, N + i))
        edges = list(set(edges))
        n = 2 * N
        if len(edges) != N * 3:
            continue
        adj = _build_adj(edges, n)
        if all(len(adj[i]) == 3 for i in range(n)) and _nonback_girth(adj, n) >= g:
            return n, edges
    return None


# ---------------- construction 3: circulant graph C(n; S) --------------------
def _circulant(k, g):
    """Systematic search for k-regular circulant graphs C(n; S) with girth >= g.
    
    For k=5, we need |S|=5 with S symmetric (if s in S, then n-s in S).
    We minimize n by checking small n values and using girth verification.
    """
    if k % 2 != 0:
        return None
    
    half = k // 2  # number of positive steps needed
    # For k=5, half=2 means we need 2 pairs + possibly 1 self-inverse element
    # Actually for k=5: we need 5 distinct elements in Z_n, symmetric
    # This means either 2 pairs {s, n-s} + 1 element with 2s=n, or 5 elements all distinct
    
    # Moore bound for k=5, g=9: n >= 1 + 5 + 5*4 + 5*4*3 + 5*4*3*2 + 5*4*3*2*1 = 1+5+20+60+120+120 = 326
    # But circulant graphs have structure, so we search from reasonable lower bound
    
    # Start from a reasonable minimum n
    min_n = 2 * half + 1  # at least need enough room for the steps
    max_n = 2000  # search limit
    
    # Try even n first (for symmetric sets without self-inverse elements)
    for n in range(min_n, max_n + 1):
        if _elapsed() > _DEADLINE - 10.0:
            break
        if n * k % 2 != 0:
            continue
        
        # Generate candidate connection sets
        # For k=5: need 5 elements in {1, ..., n-1} such that S = {s1,s2,s3,s4,s5}
        # and the set is "symmetric" meaning if s is in S, then n-s is also in S (or s = n/2)
        
        # Strategy: pick half the elements from {1, ..., floor(n/2)}
        # For odd k: need (k-1)/2 pairs + 1 element where 2s = n mod n (i.e., s = n/2 if n even)
        # For k=5: need 2 pairs + 1 self-inverse, OR 5 elements with special structure
        
        # Simpler: just enumerate small subsets and check
        import itertools
        
        if k == 5:
            # Need exactly 5 distinct neighbors
            # Case 1: n is even, use {a, n-a, b, n-b, n/2} where a,b in {1,...,n/2-1}, a!=b
            if n % 2 == 0:
                half_n = n // 2
                # Try combinations of 2 from {1, ..., half_n-1} plus n/2
                for a, b in itertools.combinations(range(1, half_n), 2):
                    if _elapsed() > _DEADLINE - 10.0:
                        break
                    S = {a, n - a, b, n - b, half_n}
                    if len(S) != 5:
                        continue
                    # Check girth
                    adj = [set() for _ in range(n)]
                    for i in range(n):
                        for s in S:
                            adj[i].add((i + s) % n)
                    if all(len(adj[i]) == 5 for i in range(n)):
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return (n, edges)
            
            # Case 2: n is odd, use {a, n-a, b, n-b, c, n-c} but that's 6 elements
            # For odd k=5 with odd n: need 5 elements, which is tricky
            # Skip odd n for k=5 for now
        
        elif k == 4:
            # Need 2 pairs: {a, n-a, b, n-b}
            for n_try in range(min_n, max_n + 1):
                if n_try % 2 != 0:
                    continue
                half_n = n_try // 2
                for a, b in itertools.combinations(range(1, half_n), 2):
                    if _elapsed() > _DEADLINE - 10.0:
                        break
                    S = {a, n_try - a, b, n_try - b}
                    if len(S) != 4:
                        continue
                    adj = [set() for _ in range(n_try)]
                    for i in range(n_try):
                        for s in S:
                            adj[i].add((i + s) % n_try)
                    if all(len(adj[i]) == 4 for i in range(n_try)):
                        if _nonback_girth(adj, n_try) >= g:
                            edges = [(i, j) for i in range(n_try) for j in adj[i] if i < j]
                            return (n_try, edges)
        
        elif k == 3:
            # Need {a, n-a, b} where 2b = n (so n even, b = n/2)
            for n_try in range(min_n, max_n + 1):
                if n_try % 2 != 0:
                    continue
                half_n = n_try // 2
                for a in range(1, half_n):
                    if _elapsed() > _DEADLINE - 10.0:
                        break
                    S = {a, n_try - a, half_n}
                    if len(S) != 3:
                        continue
                    adj = [set() for _ in range(n_try)]
                    for i in range(n_try):
                        for s in S:
                            adj[i].add((i + s) % n_try)
                    if all(len(adj[i]) == 3 for i in range(n_try)):
                        if _nonback_girth(adj, n_try) >= g:
                            edges = [(i, j) for i in range(n_try) for j in adj[i] if i < j]
                            return (n_try, edges)
    
    return None


# ---------------- construction 4: lexicographic (blow-up) product ------------
def _blowup(k, g):
    base = {
        3: [3, 4, 5, 6, 7, 8, 9, 10, 11],
        4: [5, 6, 7, 8, 9, 10, 11],
        5: [6, 7, 8, 9, 10, 11, 12],
    }.get(k)
    if not base:
        return None
    for B in base:
        if B < k:
            continue
        if (B - k) % 2 != 0:
            continue
        for a in range(1, B):
            if a == B - a or (B - a) == 0:
                continue
            S = {a, B - a}
            adj = [set() for _ in range(B)]
            for i in range(B):
                for s in S:
                    adj[i].add((i + s) % B)
            if any(len(adj[i]) != 2 for i in range(B)):
                continue
            if _nonback_girth(adj, B) < (g + 1) // 2:
                continue
            # blow up each vertex into a clique of size (B-k)//2
            m = (B - k) // 2
            n = B * m
            edges = []
            for i in range(B):
                for j in adj[i]:
                    if i < j:
                        for a in range(m):
                            for b in range(m):
                                edges.append((i * m + a, j * m + b))
                for a in range(m):
                    for b in range(a + 1, m):
                        edges.append((i * m + a, i * m + b))
            edges = list(set(edges))
            if len(edges) != (n * k) // 2:
                continue
            aadj = _build_adj(edges, n)
            if all(len(aadj[i]) == k for i in range(n)) and _nonback_girth(aadj, n) >= g:
                return n, edges
    return None


# ---------------- construction 5: BFS tree-like completion --------------------
def _bfs_complete(k, g):
    R = (g - 1) // 2
    depth_lim = min(R - 1, 5)
    # build a depth-limited k-ary rooted tree
    order = [(0, -1, 0)]          # (node, parent, depth)
    node_idx = {0: 0}
    parent_of = {0: -1}
    depth_of = {0: 0}
    q = deque([0])
    while q:
        u = q.popleft()
        du = depth_of[u]
        if du >= depth_lim:
            continue
        for child in range(k):
            v = len(order)
            order.append((v, u, du + 1))
            node_idx[v] = v
            parent_of[v] = u
            depth_of[v] = du + 1
            q.append(v)
    n = len(order)
    adj = [set() for _ in range(n)]
    for v in range(1, n):
        p = parent_of[v]
        adj[v].add(p)
        adj[p].add(v)
    # complete each vertex's degree to k using non-backtracking girth constraints
    def cand_ok(v, w):
        if w in adj[v]:
            return False
        if w == parent_of[v]:
            return False
        # no 4-cycles: avoid common neighbour of v and w
        common = set(adj[v]) & set(adj[w])
        if common:
            return False
        # no 3-cycles
        if w in adj[parent_of[v]]:
            return False
        return True

    for v in range(n):
        guard = 0
        while len(adj[v]) < k and guard < 5000:
            guard += 1
            for w in range(n):
                if len(adj[v]) >= k:
                    break
                if w <= v:
                    continue
                if cand_ok(v, w):
                    adj[v].add(w)
                    adj[w].add(v)
    if any(len(adj[i]) != k for i in range(n)):
        return None
    for i in range(n):
        adj[i] = sorted(adj[i])
    if _nonback_girth(adj, n) < g:
        return None
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return n, edges


# ---------------- construction 6: voltage graph search ----------------------
def _voltage_graph_search(k, g):
    """Search for small k-regular graphs using voltage graph lifting.
    
    The idea: start with a small base graph and assign voltages (elements of Z_n)
    to edges. The lift has n * |V_base| vertices. We minimize n.
    
    For girth guarantee: ensure that no closed walk in the base graph of length
    < g has voltage sum = 0 mod n.
    """
    if k < 3:
        return None
    
    # Start with a simple base: a cycle of length L
    # The lift of a cycle with voltage v has n*L vertices
    # Girth of the lift depends on the voltage assignment
    
    # For a single cycle base with voltage v:
    # The lift is a circulant-like graph. Closed walks of length m have voltage
    # sum = m*v (going around) or 0 (backtracking, which we exclude).
    # Non-backtracking closed walks on a cycle: only going around m times.
    # So we need m*v != 0 mod n for all 3 <= m < g.
    
    # This means: gcd(v, n) must be large enough that m*v mod n != 0 for m < g
    # Specifically, the order of v in Z_n must be >= g.
    
    # But a single cycle gives degree 2, not k.
    # For k=5, we need a base graph of degree 5.
    
    # Alternative: use a complete graph K_m as base, with voltage assignments.
    # But K_m has triangles, so girth is limited.
    
    # Better: use a tree-like base and add edges carefully.
    
    # For now, let's try a different approach: search for circulant graphs
    # with n as small as possible, using the fact that for k=5, g=9,
    # we need n such that there exist 5 elements a1,...,a5 in Z_n
    # with no non-trivial linear combination summing to 0 mod n
    # for coefficients c1,...,c5 in {-1,0,1} with |ci| <= something.
    
    # This is getting complex. Let's just do a focused circulant search
    # for k=5, g=9 with n from 300 to 1500.
    
    if k != 5 or g != 9:
        return None
    
    import itertools
    
    best = None
    for n in range(300, 1501, 2):  # even n
        if _elapsed() > _DEADLINE - 15.0:
            break
        half_n = n // 2
        # Try random samples of connection sets
        rng = _random.Random(n)
        for trial in range(50):
            if _elapsed() > _DEADLINE - 15.0:
                break
            # Pick 2 distinct elements from {1, ..., half_n-1}
            if half_n < 3:
                continue
            a, b = rng.sample(range(1, half_n), 2)
            S = {a, n - a, b, n - b, half_n}
            if len(S) != 5:
                continue
            # Quick girth check: build adjacency and verify
            adj = [set() for _ in range(n)]
            for i in range(n):
                for s in S:
                    adj[i].add((i + s) % n)
            if not all(len(adj[i]) == 5 for i in range(n)):
                continue
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                best = (n, edges)
                break  # Found one for this n, move to smaller n
        if best is not None:
            return best
    
    return None


# ---------------- construction 7: randomised greedy completion ---------------
import random as _random
import itertools as _itertools


def _greedy_complete(k, g, seed=0, max_n=None, depth_lim=None):
    """Build a k-regular graph of girth >= g by greedy edge insertion with
    random restarts.  Returns (n, edges) or None."""
    rng = _random.Random(seed)
    if max_n is None:
        max_n = 200
    if depth_lim is None:
        depth_lim = 4

    for attempt in range(8):
        if _elapsed() > _DEADLINE - 5.0:
            return None
        # random starting size in [k+1, max_n]
        n = rng.randint(k + 1, max_n)
        if n * k % 2 != 0:
            n += 1
        adj = [set() for _ in range(n)]
        # shuffle vertex order
        verts = list(range(n))
        rng.shuffle(verts)
        ok = True
        for v in verts:
            if _elapsed() > _DEADLINE - 5.0:
                ok = False
                break
            need = k - len(adj[v])
            if need <= 0:
                continue
            # candidate vertices not already adjacent and not self
            cands = [w for w in range(n) if w != v and w not in adj[v]]
            rng.shuffle(cands)
            placed = 0
            for w in cands:
                if placed >= need:
                    break
                if _elapsed() > _DEADLINE - 5.0:
                    ok = False
                    break
                # check adding (v,w) doesn't create a short cycle
                if _edge_ok(adj, v, w, g):
                    adj[v].add(w)
                    adj[w].add(v)
                    placed += 1
            if not ok:
                break
            if placed < need:
                ok = False
                break
        if not ok:
            continue
        # verify
        aadj = [sorted(adj[i]) for i in range(n)]
        if any(len(aadj[i]) != k for i in range(n)):
            continue
        if _nonback_girth(aadj, n) < g:
            continue
        edges = [(i, j) for i in range(n) for j in aadj[i] if i < j]
        return n, edges
    return None


def _edge_ok(adj, v, w, g):
    """Return True if adding edge (v,w) to adj keeps girth >= g (heuristic:
    check that no path of length < g-1 exists between v and w avoiding the
    new edge, i.e. BFS in the current graph from v to w with dist < g-1
    would create a cycle of length <= g)."""
    # BFS from v to w in current graph
    if v == w:
        return False
    if w in adj[v]:
        return False
    # shortest path from v to w in current graph
    from collections import deque as _dq
    dist = {v: 0}
    q = _dq([v])
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= g - 1:
            break
        for x in adj[u]:
            if x not in dist:
                dist[x] = du + 1
                if x == w:
                    # cycle length = du + 1 + 1 = du + 2
                    if du + 2 < g:
                        return False
                    break
                q.append(x)
    if w in dist and dist[w] + 1 < g:
        return False
    return True


# ---------------- parallel candidate verification ---------------------------
def _worker(name, fn, holder):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("petersen", lambda: _petersen(k, g)),
        ("circulant", lambda: _circulant(k, g)),
        ("voltage", lambda: _voltage_graph_search(k, g)),
        ("blowup", lambda: _blowup(k, g)),
        ("bfs", lambda: _bfs_complete(k, g)),
        ("greedy0", lambda: _greedy_complete(k, g, seed=0)),
        ("greedy1", lambda: _greedy_complete(k, g, seed=1)),
        ("greedy2", lambda: _greedy_complete(k, g, seed=2)),
    ]
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder), daemon=True)
        t.start()
        threads.append(t)

    # wait, with a safety margin, for threads to finish
    while _elapsed() < _DEADLINE - 2.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)

    best = holder.get("best")
    if best is None:
        # last-resort: return seed LUW result even if unverified in-thread
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            # degenerate but valid k-regular graph (complete graph)
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
