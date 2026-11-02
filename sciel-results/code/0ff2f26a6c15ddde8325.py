"""Cage solver: LUW construction + optimized girth check + greedy random completion."""
import time
import threading
import random
from collections import deque

_DEADLINE = 115.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle via non-backtracking walks from every vertex."""
    # Map (prev, curr) -> (source, dist)
    # Use a dict for O(1) lookup
    visited = {}
    q = deque()
    
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            visited[key] = (u, 1)
            q.append((p, u, u, 1))
    
    while q:
        v, p, u, dist = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            if w == u:
                return dist + 1
            key = (w, v)
            if key in visited:
                continue
            visited[key] = (v, dist + 1)
            q.append((w, v, u, dist + 1))
    
    return 10 ** 9


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(adj[i]) != len(set(adj[i])):
            return False
        for j in adj[i]:
            if j < 0 or j >= n or j == i:
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


# ---------------- construction: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    """LUW construction for odd k. Returns (n, edges)."""
    q = k
    # d is the dimension; girth = 2d + 1 for LUW
    # We want girth >= g, so 2d+1 >= g => d >= (g-1)/2
    d = max(3, (g - 1) // 2)
    if (2 * d + 1) < g:
        d += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build the rule for LUW
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges


# ---------------- Greedy random construction with cycle avoidance -----------
def _edge_creates_short_cycle(adj, v, u, g, max_depth=None):
    """Check if adding edge (v,u) would create a cycle of length < g."""
    if g <= 3:
        return False
    if max_depth is None:
        max_depth = g - 2
    
    # BFS from v to see if u is reachable within max_depth steps
    visited = {v: 0}
    q = deque([(v, 0)])
    while q:
        node, dist = q.popleft()
        if dist >= max_depth:
            continue
        if node == u:
            return dist + 1 < g
        for nb in adj[node]:
            if nb == u:
                if dist + 2 < g:
                    return True
                continue
            if nb not in visited:
                visited[nb] = dist + 1
                q.append((nb, dist + 1))
    return False


def _greedy_random(k, g, max_n=1300, n_restarts=30, seed=None):
    """Randomized greedy construction."""
    if seed is not None:
        random.seed(seed)
    
    best_result = None
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 12.0:
            break
        
        # Target n: start near Moore bound, go up
        R = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(R))
        target_n = max(k + 1, moore + random.randint(0, 200))
        target_n = min(target_n, max_n)
        if k % 2 == 0 and target_n % 2 != 0:
            target_n += 1
        if k % 2 == 1 and target_n % 2 == 0:
            target_n += 1
        
        adj = [set() for _ in range(target_n)]
        degree = [0] * target_n
        
        order = list(range(target_n))
        random.shuffle(order)
        
        success = True
        for v in order:
            candidates = [u for u in range(target_n) if u != v and u not in adj[v] and degree[u] < k]
            random.shuffle(candidates)
            
            added = 0
            for u in candidates:
                if added >= k:
                    break
                if _edge_creates_short_cycle(adj, v, u, g):
                    continue
                adj[v].add(u)
                adj[u].add(v)
                degree[v] += 1
                degree[u] += 1
                added += 1
            
            if added < k:
                success = False
                break
        
        if not success:
            continue
        
        if any(len(adj[i]) != k for i in range(target_n)):
            continue
        
        girth = _nonback_girth(adj, target_n)
        if girth >= g:
            edge_list = [(i, j) for i in range(target_n) for j in adj[i] if i < j]
            if best_result is None or target_n < best_result[0]:
                best_result = (target_n, edge_list)
                if _elapsed() > _DEADLINE - 18.0:
                    break
    
    return best_result


# ---------------- Local search / edge swap to reduce n -----------------------
def _try_reduce_n(k, g, n, edges, max_time=10.0):
    """Try to reduce n by removing vertices and re-wiring. Returns (n, edges) or None."""
    start = time.time()
    adj = _build_adj(edges, n)
    
    for _ in range(50):
        if time.time() - start > max_time:
            break
        if _elapsed() > _DEADLINE - 10.0:
            break
        
        # Try to remove a random vertex and reconnect its neighbors
        v = random.randrange(n)
        neighbors = list(adj[v])
        if len(neighbors) != k:
            continue
        
        # Remove v
        for u in neighbors:
            adj[u].discard(v)
        adj[v] = set()
        
        # Try to reconnect: for each pair of neighbors, add edge if it doesn't create short cycle
        # We need to maintain k-regularity, so we need to add k/2 edges among neighbors
        # (since each neighbor lost one edge to v)
        # Actually each neighbor needs one more edge. Pair them up.
        random.shuffle(neighbors)
        paired = False
        if k % 2 == 0:
            for i in range(0, k, 2):
                a, b = neighbors[i], neighbors[i + 1]
                if a == b or b in adj[a]:
                    paired = False
                    break
                if _edge_creates_short_cycle(adj, a, b, g):
                    paired = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if not paired:
                # Undo
                for u in neighbors:
                    adj[u].discard(neighbors[(neighbors.index(u) + 1) % k] if False else None)
                # Just undo by removing what we added
                for i in range(0, k, 2):
                    a, b = neighbors[i], neighbors[i + 1]
                    adj[a].discard(b)
                    adj[b].discard(a)
                continue
        else:
            # Odd k: can't pair up. Skip for now.
            for u in neighbors:
                adj[u].discard(v)
            adj[v] = set()
            # Restore
            for u in neighbors:
                adj[u].add(v)
            adj[v] = set(neighbors)
            continue
        
        # Verify
        if all(len(adj[i]) == k for i in range(n - 1)):
            new_n = n - 1
            new_edges = [(i, j) for i in range(new_n) for j in adj[i] if i < j]
            # Quick girth check
            if _nonback_girth(adj, new_n) >= g:
                return new_n, new_edges
            else:
                # Undo
                for u in neighbors:
                    adj[u].discard(neighbors[(neighbors.index(u) + 1) % k] if False else None)
                for i in range(0, k, 2):
                    a, b = neighbors[i], neighbors[i + 1]
                    adj[a].discard(b)
                    adj[b].discard(a)
    
    return None


def _worker(name, fn, holder, lock):
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
        adj = _build_adj(edges, n)
        if n > 0 and _is_valid(adj, n, holder.get("k", 5), holder.get("g", 9)):
            with lock:
                cur = holder.get("best")
                if cur is None or n < cur[0]:
                    holder["best"] = (n, edges)
    except Exception:
        pass


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None, "k": k, "g": g}
    lock = threading.Lock()
    
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("greedy", lambda: _greedy_random(k, g, seed=42)),
    ]
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, lock), daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for threads
    while _elapsed() < _DEADLINE - 5.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)
    
    for t in threads:
        t.join(timeout=max(0.1, _DEADLINE - _elapsed() - 1.0))
    
    with lock:
        best = holder.get("best")
    
    if best is None:
        # Fallback: LUW
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    
    # Try to reduce n with local search if we have time
    if _elapsed() < _DEADLINE - 15.0:
        try:
            reduced = _try_reduce_n(k, g, best[0], best[1], max_time=min(10.0, _DEADLINE - _elapsed() - 5.0))
            if reduced is not None:
                adj = _build_adj(reduced[1], reduced[0])
                if _is_valid(adj, reduced[0], k, g):
                    best = reduced
        except Exception:
            pass
    
    return best
