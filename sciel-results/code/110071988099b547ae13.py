"""Combined cage solver: Finite field constructions, Circulant search, Random Configuration Model, and LUW fallback."""
import time
import threading
import random
from collections import deque
from math import isqrt

_DEADLINE = 118.0
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


def _has_short_cycle(adj, n, g):
    """Fast check: does the graph have any cycle of length < g?
    Uses BFS from each vertex up to depth g-2, tracking (vertex, parent) to avoid backtracking."""
    max_depth = g - 2  # cycles of length < g have radius < g/2, but we check up to g-2 to be safe
    # Actually, a cycle of length L means two paths of length L/2 from some vertex meet.
    # Better: for each directed edge (u->v), do non-backtracking BFS from v up to depth g-2.
    # If we reach u, we found a cycle of length <= g-1.
    for u in range(n):
        for v in adj[u]:
            # BFS from v, not going back to u, up to depth g-2
            visited = {}  # (node, parent) -> dist
            q = deque()
            for w in adj[v]:
                if w != u:
                    visited[(w, v)] = 1
                    q.append((w, v))
            found = False
            while q:
                (node, parent), dist = q.popleft()
                if dist >= g - 1:
                    continue
                if node == u:
                    return True
                for w in adj[node]:
                    if w == parent:
                        continue
                    if w == u:
                        return True
                    key = (w, node)
                    if key in visited:
                        continue
                    visited[key] = dist + 1
                    q.append((w, node))
    return False


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


# ---------------- construction 1: Finite field (incidence graph) -------------
def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime_ge(n):
    if n <= 2:
        return 2
    if n <= 3:
        return 3
    p = n if n % 2 == 1 else n + 1
    while not _is_prime(p):
        p += 2
    return p


def _field_construction(k, g):
    """Deterministic construction from finite fields.
    
    For q a prime power with q+1 = k, the incidence graph of the generalized
    quadrangle GQ(q, q) is (q+1)-regular with girth 2*ceil(log_{q+1}(q^3+q^2+q+1))
    or more precisely girth = 2*ceil(3*log_q(q)) ... actually girth is 2*ceil(log_{q+1}(q^3+1))
    but for GQ(q,q) the girth is 2*ceil(3) = 6? No.
    
    Actually, the incidence graph of GQ(q,q) has girth 2*ceil(log_{q+1}(q^3+q^2+q+1))...
    Let me think more carefully.
    
    The generalized quadrangle GQ(q,q) has:
    - Points: (q+1)^3 * (q+1) = (q+1)^4 points? No.
    - GQ(q,q) has (q+1)^3(q+1) = (q+1)^4 points and same number of lines.
    - Each point is on q+1 lines, each line has q+1 points.
    - The incidence graph is (q+1)-regular.
    - Number of vertices in incidence graph: 2*(q+1)^4
    - Girth: The girth of the incidence graph of GQ(q,q) is 2*ceil(3*log_{q+1}(q+1)) = 6? 
      No, the girth is determined by the parameter. For GQ(s,t), the girth of the incidence graph is 2*ceil(log_{s+1}(st+1))... 
    
    Actually, for GQ(q,q): the incidence graph has girth 2*ceil(log_{q+1}(q^2+q+1)) + 1? 
    Let me just compute it directly for small q.
    
    For q=2: GQ(2,2) has 27 points, 27 lines, 54 vertices, 5-regular. Girth should be 8.
    For q=3: GQ(3,3) has 64 points, 64 lines, 128 vertices, 4-regular. 
    For q=4: GQ(4,4) has 125 points, 125 lines, 250 vertices, 5-regular. Girth = 12.
    
    Wait, that's not matching. Let me reconsider.
    
    GQ(s,t) has (s+1)^2(s^2+t) points... no.
    
    Standard GQ(s,t): 
    - Number of points: (s+1)^2 * (s^2 + t) ... no.
    - Actually: points = (s+1)^2(s^2+t), lines = (t+1)^2(t^2+s)
    - For GQ(q,q): points = (q+1)^2(q^2+q) = q(q+1)^3, lines = same.
    - Each point on t+1 = q+1 lines, each line has s+1 = q+1 points.
    - Incidence graph: 2*q*(q+1)^3 vertices, (q+1)-regular.
    
    For q=4: 2*4*5^3 = 2*4*125 = 1000 vertices. That's too many.
    
    Hmm, let me reconsider. The directive says "incidence graph of the unique generalized hexagon of order (4,4), which is a known 5-regular graph with girth 12 and exactly 330 vertices."
    
    A generalized hexagon of order (s,t) has girth 12 (it's a generalized polygon with 6 sides, so girth = 2*6 = 12).
    
    The unique generalized hexagon of order (4,4) has:
    - Points: (s+1)^2(s^2+st+t^2) = 5^2*(16+16+16) = 25*48 = 1200? No.
    
    Actually for a generalized polygon of order (s,t) with n sides:
    - Points: (s+1)^n * (s^{n-1} + ... ) ... this is getting complicated.
    
    Let me just use the explicit construction from F_5 for the specific case k=5, g<=12.
    
    Construction: Vertices are pairs (side, x) where side in {0,1} and x in F_5^2.
    Total vertices: 2 * 25 = 50. That's not 330 either.
    
    Let me think about this differently. The Moore bound for k=5, g=9 is:
    R = 4, moore = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426.
    
    The best known is 1116 vertices. The directive claims 330 vertices for girth 12, which would be excellent if correct.
    
    Let me try a different approach: use the construction from the paper by Exoo et al. or a known explicit construction.
    
    Actually, let me try: vertices = F_5^3 x {0,1}, so 2*125 = 250 vertices. Still not 330.
    
    Or maybe F_5^2 x F_5 x {0,1} = 2*125 = 250.
    
    330 = 2 * 165 = 2 * 3 * 5 * 11. Hmm.
    
    Let me try a different structural approach: use a circulant-like construction with a more careful search, or use the Wagner construction.
    
    For now, let me implement a clean finite-field-based construction that works for k = p+1 where p is prime, using the incidence graph of a symmetric design or similar structure.
    """
    # For k = q+1 where q is prime, use a construction based on F_q
    # Simple construction: vertices are (i, a) where i in range(k), a in F_q
    # This gives k*q vertices, but we need k-regular.
    
    # Alternative: Use the Paley graph or similar for specific cases.
    # For k=5: try to construct using F_5
    # Vertices: elements of F_5 x F_5 = 25 vertices, but we need 5-regular, so 25 vertices with degree 5 means 62.5 edges - not integer.
    
    # Let me try: n = k*q where q is chosen appropriately
    # For k=5, q=5: n=25, need 5-regular -> 62.5 edges. Not valid.
    # For k=5, q=6: n=30, need 5-regular -> 75 edges. Valid.
    
    # Actually, let me just implement a robust random + local search approach that's faster,
    # and keep the LUW as fallback. The key insight from the directive is to try a new family.
    
    # New family: Use algebraic construction with F_p for p = k-1 (if prime)
    p = k - 1
    if not _is_prime(p):
        return None
    
    # Vertices: (side, x, y) where side in {0,1}, x,y in F_p
    # Total: 2 * p * p vertices
    n = 2 * p * p
    if n < k + 1:
        return None
    
    adj = [set() for _ in range(n)]
    
    def idx(side, x, y):
        return side * p * p + x * p + y
    
    for side in range(2):
        for x in range(p):
            for y in range(p):
                u = idx(side, x, y)
                # Connect to 5 vertices on the other side
                # Use a permutation-based construction
                for t in range(p):
                    if t == 0:
                        # Special connection
                        vx = (x + y) % p
                        vy = (y - x) % p
                    else:
                        inv_t = pow(t, p - 2, p)  # t^{-1} mod p
                        vx = (x + t * y) % p
                        vy = (y + t * x) % p
                    
                    v = idx(1 - side, vx, vy)
                    adj[u].add(v)
                    adj[v].add(u)
    
    # Check if this is k-regular
    for i in range(n):
        if len(adj[i]) != k:
            return None
        if len(set(adj[i])) != len(adj[i]):
            return None
    
    # Check for loops
    for i in range(n):
        if i in adj[i]:
            return None
    
    # Check girth
    if _nonback_girth(adj, n) < g:
        return None
    
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return n, edges


# ---------------- construction 2: LUW (Lazebnik-Ustimenko-Woldar) ------------
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
    
    if k % 2 == 0:
        half = k // 2
        # Try a range of n values, prioritizing smaller n
        for n in range(moore, moore + 200):
            if n * k % 2 != 0: continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            # For small n, do exhaustive search; for larger n, sample
            if n <= 100:
                import itertools
                for S in itertools.combinations(range(1, n // 2 + 1), half):
                    S_set = set(S)
                    if any((n - s) % n in S_set for s in S):
                        continue
                    adj = [set() for _ in range(n)]
                    for i in range(n):
                        for s in S:
                            adj[i].add((i + s) % n)
                            adj[i].add((i - s) % n)
                    if all(len(adj[i]) == k for i in range(n)):
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return n, edges
            else:
                # Random sampling for larger n
                rng = random.Random(n * 1000 + k)
                for _ in range(50):
                    if _elapsed() > _DEADLINE - 15.0:
                        return None
                    S = rng.sample(range(1, n // 2 + 1), half)
                    S_set = set(S)
                    if any((n - s) % n in S_set for s in S):
                        continue
                    adj = [set() for _ in range(n)]
                    for i in range(n):
                        for s in S:
                            adj[i].add((i + s) % n)
                            adj[i].add((i - s) % n)
                    if all(len(adj[i]) == k for i in range(n)):
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return n, edges
    else:
        m = (k - 1) // 2
        for n in range(moore, moore + 200, 2):
            if n < k + 1: continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            if n <= 100:
                import itertools
                for S_vals in itertools.combinations(range(1, n // 2), m):
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
            else:
                rng = random.Random(n * 1000 + k)
                for _ in range(50):
                    if _elapsed() > _DEADLINE - 15.0:
                        return None
                    S_vals = rng.sample(range(1, n // 2), m)
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
    
    # Search n from moore to moore + 40
    for n in range(moore, moore + 40):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
            
        for attempt in range(20):
            if _elapsed() > _DEADLINE - 8.0:
                return None
            
            # Generate random k-regular graph with retries
            adj = None
            for _ in range(50):
                adj = _generate_random_regular(n, k, rng)
                if adj is not None:
                    break
            if adj is None:
                continue
                
            # Check girth
            if not _has_short_cycle(adj, n, g):
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
                
            # Local search: try to break short cycles
            max_steps = 500
            for step in range(max_steps):
                if _elapsed() > _DEADLINE - 8.0:
                    return None
                
                # Fast check for short cycles
                if not _has_short_cycle(adj, n, g):
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
                    
                # Pick a random edge (u, v)
                u = rng.randint(0, n - 1)
                v = rng.choice(list(adj[u]))
                
                # Pick another random edge (w, x) not incident to u, v
                w = -1
                x = -1
                for _ in range(50):
                    w = rng.randint(0, n - 1)
                    if w in (u, v):
                        continue
                    if w in adj[u] or w in adj[v]:
                        continue
                    if not adj[w]:
                        continue
                    x = rng.choice(list(adj[w]))
                    if x in (u, v, w):
                        continue
                    if x in adj[u] or x in adj[v]:
                        continue
                    # Check no multi-edges after swap
                    if w in adj[u] or x in adj[v]:
                        continue
                    
                    # Execute swap: remove (u,v) and (w,x), add (u,w) and (v,x)
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
    # Configuration model with rejection
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
        
    return adj


def _greedy_complete(k, g, seed=0):
    """Greedy completion: start with a partial graph and complete it while avoiding short cycles."""
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    for n in range(moore, moore + 30):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
        
        for attempt in range(10):
            if _elapsed() > _DEADLINE - 8.0:
                return None
            
            # Start with empty adjacency
            adj = [set() for _ in range(n)]
            remaining = [k] * n  # degree remaining for each vertex
            
            # Greedily add edges
            for _ in range(n * k // 2):
                if _elapsed() > _DEADLINE - 8.0:
                    return None
                
                # Find vertices with remaining degree
                candidates = [i for i in range(n) if remaining[i] > 0]
                if len(candidates) < 2:
                    break
                
                # Try random pairs until we find a valid edge
                found = False
                for _ in range(100):
                    u = rng.choice(candidates)
                    v = rng.choice(candidates)
                    if u == v or v in adj[u]:
                        continue
                    
                    # Add edge temporarily
                    adj[u].add(v)
                    adj[v].add(u)
                    
                    # Check if this creates a short cycle using fast check
                    if _has_short_cycle(adj, n, g):
                        # Remove the edge
                        adj[u].remove(v)
                        adj[v].remove(u)
                    else:
                        remaining[u] -= 1
                        remaining[v] -= 1
                        found = True
                        break
                
                if not found:
                    break
            
            # Check if we completed the graph
            if all(len(adj[i]) == k for i in range(n)):
                if not _has_short_cycle(adj, n, g):
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
    
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    
    jobs = [
        ("field", lambda: _field_construction(k, g)),
        ("circulant", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("greedy0", lambda: _greedy_complete(k, g, seed=0)),
        ("greedy1", lambda: _greedy_complete(k, g, seed=1)),
        ("luw", lambda: _luw(k, g)),
    ]
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
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


def _worker(name, fn, holder, k, g):
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
