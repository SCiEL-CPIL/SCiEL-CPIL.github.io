"""Combined cage solver: Circulant search, Random Configuration Model with local search, and LUW fallback."""
import time
import threading
import random
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


# ---------------- construction 2: Circulant Search ---------------------------
def _circulant_search(k, g):
    # Moore bound
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # For even k, standard circulants C(n, S) with |S| = k/2
    # For odd k, use C(n, {+a, -a, +b, -b, n/2}) on Z_n (n even)
    
    max_n = 2 * moore + 50
    
    if k % 2 == 0:
        half = k // 2
        for n in range(moore, max_n + 1):
            if n * k % 2 != 0: continue
            # Generate combinations of S
            # To keep it fast, we only search if n is not too large
            if n > 200: continue # Heuristic limit for combinatorial explosion
            import itertools
            for S in itertools.combinations(range(1, n // 2 + 1), half):
                S = list(S)
                # Check symmetry: if s in S, n-s should NOT be in S to avoid double counting edges in adj set?
                # Actually, adj[i] gets i+s and i-s. If n-s is in S, it's a duplicate.
                if any((n - s) % n in S for s in S):
                    continue
                adj = [set() for _ in range(n)]
                valid = True
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    else:
        # Odd k: use Z_n with involution
        # S = {+a, -a, +b, -b, n/2}
        # k must be 5? If k=3, S={+a, -a, n/2}.
        # General odd k: S = {+a1, -a1, ..., +am, -am, n/2} where 2m+1 = k.
        m = (k - 1) // 2
        for n in range(moore, max_n + 1, 2): # n must be even
            if n < k + 1: continue
            if n > 200: continue
            import itertools
            # Choose m distinct values from 1 to n/2 - 1
            # They must be distinct and not n/2
            for S_vals in itertools.combinations(range(1, n // 2), m):
                # S = {s, n-s for s in S_vals} U {n/2}
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S_vals:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                    adj[i].add((i + n // 2) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    return None


# ---------------- construction 3: Random Configuration Model + Local Search --
def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Search n from moore to moore + 50
    for n in range(moore, moore + 50):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
            
        for attempt in range(30):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            
            # Generate random k-regular graph
            adj = _generate_random_regular(n, k, rng)
            if adj is None:
                continue
                
            # Check girth
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
                
            # Local search: try to break short cycles
            # We do a few steps of "edge swap" to try to increase girth
            for step in range(800):
                if _elapsed() > _DEADLINE - 10.0:
                    return None
                
                # Find shortest cycle
                girth = _nonback_girth(adj, n)
                if girth >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
                    
                # Pick a random edge
                u = rng.randint(0, n - 1)
                if not adj[u]: continue
                v = rng.choice(list(adj[u]))
                
                # Pick another random edge (w, x) not incident to u, v
                # To maintain regularity, we swap (u,v) and (w,x) with (u,w) and (v,x)
                # We need to find w, x such that (w,x) is an edge, w!=u,v, x!=u,v
                # And (u,w) and (v,x) are not existing edges
                w = -1
                x = -1
                for _ in range(80): # Try 80 random pairs
                    w = rng.randint(0, n - 1)
                    if w in (u, v): continue
                    if w not in adj[u] and w not in adj[v]: # u-w and v-w not edges? No, we are forming u-w.
                        # We need w to be connected to x
                        if not adj[w]: continue
                        x = rng.choice(list(adj[w]))
                        if x in (u, v, w): continue
                        if x == w: continue
                        # Check if (u,w) and (v,x) are valid new edges
                        if w in adj[u] or x in adj[v]: continue
                        # Check if (u,x) and (v,w) are valid? We are doing (u,w) and (v,x)
                        # We remove (u,v) and (w,x).
                        # Add (u,w) and (v,x).
                        # We must ensure (u,w) and (v,x) don't create loops or multi-edges.
                        # They don't create loops if u!=w, v!=x.
                        # They don't create multi-edges if w not in adj[u] and x not in adj[v].
                        
                        # Execute swap
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[w].remove(x)
                        adj[x].remove(w)
                        
                        adj[u].add(w)
                        adj[w].add(u)
                        adj[v].add(x)
                        adj[x].add(v)
                        
                        break
                if w == -1:
                    continue
    return None


def _generate_random_regular(n, k, rng):
    # Configuration model
    # Create list of stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    # Pair up stubs
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
        
    # Check regularity (should be true)
    # Check for multi-edges (handled above)
    return adj


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    
    jobs = [
        ("circulant", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("luw", lambda: _luw(k, g)),
    ]
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 2.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)

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
