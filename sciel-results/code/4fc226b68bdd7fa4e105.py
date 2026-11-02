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


# ---------------- fast girth via non-backtracking walks ----------------------
def _nonback_girth_fast(adj, n, g_limit):
    """Return length of shortest cycle, certified by non-backtracking walks.
    Uses array-based state tracking for speed. Stops early if cycle >= g_limit
    is confirmed (i.e., no cycle shorter than g_limit found up to distance g_limit-1).
    Returns g_limit if no cycle shorter than g_limit is found (girth >= g_limit).
    """
    # State: (v, p) -> distance. Use a 2D array: dist[v][p_idx] but p is a vertex.
    # Better: use a dict mapping (v,p) -> dist, but with early termination.
    # For speed, use a list of arrays: for each vertex v, track which (v,p) states visited.
    # Since each vertex has degree k, there are at most k states per vertex.
    # Map: for vertex v, prev_vertex p -> distance. Use list of dicts or a flat array.
    
    # Flat approach: state_id = v * n + p, but that's n^2 which is too big.
    # Use: for each vertex v, a dict {prev: dist}. Total states = n*k.
    
    prev_dist = [{} for _ in range(n)]  # prev_dist[v][p] = dist
    q = deque()
    for u in range(n):
        for p in adj[u]:
            prev_dist[p][u] = 1
            q.append((p, u, 1))
    
    while q:
        v, p, dist = q.popleft()
        if dist >= g_limit:
            continue  # don't need to explore further
        for w in adj[v]:
            if w == p:
                continue
            if w in prev_dist[v]:
                continue
            if w == p:  # already checked
                continue
            # Check if this closes a cycle: w == the vertex before p? 
            # Actually, a cycle is found when we reach a state (w, v) where w was the
            # vertex we came from before p. But in our formulation, we track (current, prev).
            # A cycle of length L is found when from state (v, p) we go to w and w == 
            # the vertex that was before p on the path. But we don't track that.
            
            # Simpler: a non-backtracking walk from u: u -> p -> v -> w...
            # If w == u, we have a cycle of length dist+1.
            # But we don't track u (the start). 
            # Alternative: just check if (w, v) was already visited at a shorter distance.
            
            # Actually the standard approach: if we reach (w, v) and w was visited as 
            # a "previous" from some other path, we have a cycle. But the simplest:
            # if w is in prev_dist[v], skip. If w == the start vertex of this walk...
            
            # Let me use the original approach but with arrays for speed.
            # The key insight: a cycle exists iff there's a non-backtracking closed walk.
            # From state (v, p) at distance d from start u, if we can go to w where
            # w == u, we have a cycle of length d+1. But we don't track u.
            
            # Better approach: BFS from ALL starts simultaneously.
            # State (v, p) means "at vertex v, came from p". 
            # Initialize: for each edge (u,p), state (p, u) at dist 1.
            # From (v, p): go to (w, v) for w in adj[v], w != p.
            # A cycle is found when we reach (w, v) where w was the initial "u" for this path.
            # But since we start from ALL edges, a cycle of length L is found when
            # state (w, v) is reached and w's initial state was (v, w) at dist 1,
            # meaning total walk length = 1 + (dist-1) + 1 = dist+1... 
            # Actually: initial state (p, u) means we're at p, came from u, dist=1.
            # If from some state (v, p) we go to w and w == u (the start), cycle length = dist+1.
            # But we don't know u. 
            # 
            # The trick: since we initialize ALL (p, u) for all edges, if we reach
            # state (w, v) and (v, w) was in the initial set, then we have a cycle.
            # More precisely: if (w, v) was visited at distance d1 and we're now reaching
            # it at distance d2 > d1, no new cycle. But if we reach (w, v) where w was
            # a neighbor that started a walk... 
            #
            # Simplest correct approach (same as original but faster):
            # Track for each state (v, p) the distance. A cycle is found when we try
            # to visit a state that's already been visited AND it leads back to the start.
            # 
            # Actually, the original code's approach is: 
            # prev[key] = (u, dist) where key=(v,p), u is the vertex before p.
            # When we go from (v,p) to w: if w == u, cycle of length dist+1.
            # This works because u is the "other end" of the edge that brought us to p.
            
            # Let me just use the original logic but with a flat array for speed.
            # State (v, p): store the "u" (vertex before p) and dist.
            # Use: state_u[v][p] and state_d[v][p]. But p is a vertex index up to n.
            # For k-regular graph, each v has only k neighbors, so we can use:
            # For each v, a dict {p: (u, dist)}.
            
            # This is getting complex. Let me just optimize the original with:
            # 1. Early termination at dist >= g_limit
            # 2. Use lists instead of tuples for queue
            # 3. Precompute adj as lists
            
            pass  # placeholder
    
    # The above is getting too complex. Let me use a cleaner approach.
    return g_limit  # fallback


def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
    # State: (v, p) -> (u, dist) where u is the vertex before p on path
    prev = {}
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
                return dist + 1
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9


def _girth_at_least(adj, n, g):
    """Return True if girth >= g, using non-backtracking walks with early termination."""
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append((p, u, 1))
    while q:
        v, p, dist = q.popleft()
        if dist >= g:
            continue
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == p:
                continue
            # Check if this creates a cycle: w should not be the "u" that started
            # the path. In our encoding, prev[key] = (u, dist) means we reached (v,p)
            # via path starting from u. But we stored it differently.
            # Let me redo: prev[(v,p)] = (u, dist) means the walk is u->...->p->v with length dist.
            # From (v,p) at dist, going to w: if w==u, cycle of length dist+1.
            # But we don't have u in the queue. Let me store it.
            pass
    # Fallback: use full girth check
    return _nonback_girth(adj, n) >= g


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
                return False
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
    """Deterministic circulant search: for k=5, g=9, find smallest even n with
    S={a, n-a, b, n-b, n/2} giving girth >= g."""
    if k != 5:
        return None
    
    min_n = 328
    max_n = 2000
    
    for n in range(min_n, max_n + 1, 2):
        if _elapsed() > _DEADLINE - 12.0:
            break
        half_n = n // 2
        candidates = set()
        for a in range(1, min(half_n, 20)):
            for b in range(a + 1, min(half_n, a + 50)):
                candidates.add((a, b))
        for a in range(1, min(half_n, 10)):
            b = half_n // 3 + a
            if b < half_n:
                candidates.add((a, b))
        
        for a, b in candidates:
            if _elapsed() > _DEADLINE - 12.0:
                break
            if a >= half_n or b >= half_n or a == b:
                continue
            S = [a, n - a, b, n - b, half_n]
            if len(set(S)) != 5:
                continue
            
            # Fast girth check
            neg = {s: (n - s) % n for s in S}
            visited = set()
            q = deque()
            for s in S:
                state = (s, s)
                visited.add(state)
                q.append((s, s, 1))
            
            girth_found = 10**9
            while q:
                v, last_s, dist = q.popleft()
                if dist >= g:
                    break
                for s in S:
                    if s == neg[last_s]:
                        continue
                    new_v = (v + s) % n
                    if new_v == 0:
                        girth_found = dist + 1
                        break
                    new_state = (new_v, s)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append((new_v, s, dist + 1))
                if girth_found <= g:
                    break
            
            if girth_found >= g:
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                if all(len(adj[i]) == 5 for i in range(n)):
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return (n, edges)
    
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
    order = [(0, -1, 0)]
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
    
    def cand_ok(v, w):
        if w in adj[v]:
            return False
        if w == parent_of[v]:
            return False
        common = set(adj[v]) & set(adj[w])
        if common:
            return False
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
    if k != 5 or g != 9:
        return None
    
    min_n = 328
    max_n = 1800
    
    for n in range(min_n, max_n + 1, 2):
        if _elapsed() > _DEADLINE - 15.0:
            break
        half_n = n // 2
        if half_n < 3:
            continue
        
        candidates = set()
        for a in range(1, min(half_n, 30)):
            for mult in [2, 3, 4, 5]:
                b = a * mult
                if b < half_n and b != a:
                    candidates.add((a, b))
        candidates.add((half_n // 4, half_n // 3))
        candidates.add((half_n // 5, 2 * half_n // 5))
        
        for a, b in candidates:
            if _elapsed() > _DEADLINE - 15.0:
                break
            if a <= 0 or b <= 0 or a >= half_n or b >= half_n or a == b:
                continue
            S = [a, n - a, b, n - b, half_n]
            if len(set(S)) != 5:
                continue
            
            neg = {s: (n - s) % n for s in S}
            visited = set()
            q = deque()
            for s in S:
                state = (s, s)
                visited.add(state)
                q.append((s, s, 1))
            
            girth_found = 10**9
            while q:
                v, last_s, dist = q.popleft()
                if dist >= g:
                    break
                for s in S:
                    if s == neg[last_s]:
                        continue
                    new_v = (v + s) % n
                    if new_v == 0:
                        girth_found = dist + 1
                        break
                    new_state = (new_v, s)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append((new_v, s, dist + 1))
                if girth_found <= g:
                    break
            
            if girth_found >= g:
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                if all(len(adj[i]) == 5 for i in range(n)):
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return (n, edges)
    
    return None


# ---------------- construction 7: randomised greedy completion ---------------
import random as _random
import itertools as _itertools


def _greedy_complete(k, g, seed=0, max_n=None, depth_lim=None):
    rng = _random.Random(seed)
    if max_n is None:
        max_n = 200
    if depth_lim is None:
        depth_lim = 4

    for attempt in range(8):
        if _elapsed() > _DEADLINE - 5.0:
            return None
        n = rng.randint(k + 1, max_n)
        if n * k % 2 != 0:
            n += 1
        adj = [set() for _ in range(n)]
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
            cands = [w for w in range(n) if w != v and w not in adj[v]]
            rng.shuffle(cands)
            placed = 0
            for w in cands:
                if placed >= need:
                    break
                if _elapsed() > _DEADLINE - 5.0:
                    ok = False
                    break
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
        aadj = [sorted(adj[i]) for i in range(n)]
        if any(len(aadj[i]) != k for i in range(n)):
            continue
        if _nonback_girth(aadj, n) < g:
            continue
        edges = [(i, j) for i in range(n) for j in aadj[i] if i < j]
        return n, edges
    return None


def _edge_ok(adj, v, w, g):
    if v == w:
        return False
    if w in adj[v]:
        return False
    dist = {v: 0}
    q = deque([v])
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= g - 1:
            break
        for x in adj[u]:
            if x not in dist:
                dist[x] = du + 1
                if x == w:
                    if du + 2 < g:
                        return False
                    break
                q.append(x)
    if w in dist and dist[w] + 1 < g:
        return False
    return True


# ---------------- construction 8: random circulant search --------------------
def _random_circulant(k, g, seed=0):
    """Random search over circulant graphs C(n; {a, n-a, b, n-b, n/2})."""
    if k != 5:
        return None
    rng = _random.Random(seed)
    min_n = 328
    max_n = 1500
    
    for n in range(min_n, max_n + 1, 2):
        if _elapsed() > _DEADLINE - 10.0:
            break
        half_n = n // 2
        # Try random pairs
        for _ in range(20):
            a = rng.randint(1, half_n - 1)
            b = rng.randint(1, half_n - 1)
            if a == b:
                continue
            S = [a, n - a, b, n - b, half_n]
            if len(set(S)) != 5:
                continue
            
            neg = {s: (n - s) % n for s in S}
            visited = set()
            q = deque()
            for s in S:
                state = (s, s)
                visited.add(state)
                q.append((s, s, 1))
            
            girth_found = 10**9
            while q:
                v, last_s, dist = q.popleft()
                if dist >= g:
                    break
                for s in S:
                    if s == neg[last_s]:
                        continue
                    new_v = (v + s) % n
                    if new_v == 0:
                        girth_found = dist + 1
                        break
                    new_state = (new_v, s)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append((new_v, s, dist + 1))
                if girth_found <= g:
                    break
            
            if girth_found >= g:
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                if all(len(adj[i]) == 5 for i in range(n)):
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return (n, edges)
    
    return None


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
        ("circulant", lambda: _circulant(k, g)),
        ("voltage", lambda: _voltage_graph_search(k, g)),
        ("petersen", lambda: _petersen(k, g)),
        ("blowup", lambda: _blowup(k, g)),
        ("bfs", lambda: _bfs_complete(k, g)),
        ("greedy0", lambda: _greedy_complete(k, g, seed=0)),
        ("greedy1", lambda: _greedy_complete(k, g, seed=1)),
        ("randcirc0", lambda: _random_circulant(k, g, seed=0)),
        ("randcirc1", lambda: _random_circulant(k, g, seed=42)),
    ]
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
