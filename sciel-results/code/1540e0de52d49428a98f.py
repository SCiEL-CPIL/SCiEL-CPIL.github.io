"""Improved cage solver: Optimized Circulant search and Greedy Local Search."""
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


# ---------------- construction 2: Circulant Search (Optimized) ---------------------------
def _circulant_search(k, g):
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # For even k, standard circulants C(n, S) with |S| = k/2
    # For odd k, use C(n, {+a, -a, +b, -b, n/2}) on Z_n (n even)
    
    max_n = 2 * moore + 200 # Slightly expanded range
    
    if k % 2 == 0:
        half = k // 2
        import itertools
        for n in range(moore, max_n + 1):
            if n * k % 2 != 0: continue
            if _elapsed() > _DEADLINE - 10.0:
                return None
            
            # To keep it fast, we sample combinations if n is large
            # For small n, we do exhaustive
            limit = 10000
            count = 0
            for S in itertools.combinations(range(1, n // 2 + 1), half):
                count += 1
                if count > limit:
                    break
                if _elapsed() > _DEADLINE - 10.0:
                    return None
                S = list(S)
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
        m = (k - 1) // 2
        import itertools
        for n in range(moore, max_n + 1, 2):
            if n < k + 1: continue
            if _elapsed() > _DEADLINE - 10.0:
                return None
            limit = 10000
            count = 0
            for S_vals in itertools.combinations(range(1, n // 2), m):
                count += 1
                if count > limit:
                    break
                if _elapsed() > _DEADLINE - 10.0:
                    return None
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


# ---------------- construction 3: Greedy Local Search ---------------------------
def _local_search(k, g, seed=0):
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Try to find a graph with n close to moore
    # We search a range of n values
    max_n = moore + 100
    
    for n in range(moore, max_n + 1):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
        
        for attempt in range(10):
            if _elapsed() > _DEADLINE - 5.0:
                return None
            
            adj = _generate_random_regular(n, k, rng)
            if adj is None:
                continue
            
            current_girth = _nonback_girth(adj, n)
            
            # If already valid, return
            if current_girth >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            
            # Local Search: Greedy Cycle Breaking
            # We try to increase girth by swapping edges in short cycles
            for step in range(1000):
                if _elapsed() > _DEADLINE - 5.0:
                    return None
                
                # Find shortest cycle
                girth = _nonback_girth(adj, n)
                if girth >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
                
                # Find a specific short cycle to break
                # We pick a random vertex and do a BFS to find a cycle
                # For simplicity, we just pick a random edge and try to swap it
                # This is a heuristic.
                
                # Better: Find the shortest cycle using BFS from a random start
                start = rng.randint(0, n - 1)
                cycle = _find_short_cycle(adj, n, start)
                
                if not cycle:
                    continue
                
                # Try to break the cycle by swapping one of its edges
                # Cycle is a list of vertices [v0, v1, ..., vL, v0]
                # Edges are (v0,v1), (v1,v2), ...
                
                swapped = False
                for i in range(len(cycle) - 1):
                    u = cycle[i]
                    v = cycle[i + 1]
                    # Try to swap (u, v) with (w, x)
                    # We need w, x such that (w,x) is an edge, and (u,w), (v,x) are not edges
                    # And w, x not in cycle (ideally)
                    
                    # Pick random w from adj[u] \ {v, cycle[i-1], cycle[i+1]}
                    # Pick random x from adj[w] \ {u, ...}
                    
                    candidates_u = [x for x in adj[u] if x != v and x != cycle[i-1] and x != cycle[(i+1)%len(cycle)]]
                    if not candidates_u:
                        continue
                    
                    w = rng.choice(candidates_u)
                    
                    # Now find x in adj[w] such that (v,x) is not an edge
                    candidates_x = [x for x in adj[w] if x != u and x != v]
                    if not candidates_x:
                        continue
                    
                    x = rng.choice(candidates_x)
                    
                    # Check if (v, x) is already an edge
                    if x in adj[v]:
                        continue
                    
                    # Check if (u, x) or (v, w) are edges?
                    # We are adding (u, w) and (v, x).
                    # We are removing (u, v) and (w, x).
                    # We must ensure (u, w) is not an edge (it isn't, w in candidates_u)
                    # We must ensure (v, x) is not an edge (checked)
                    
                    # Perform swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[w].remove(x)
                    adj[x].remove(w)
                    
                    adj[u].add(w)
                    adj[w].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    
                    # Check if girth improved or stayed same
                    new_girth = _nonback_girth(adj, n)
                    if new_girth < current_girth:
                        # Revert
                        adj[u].remove(w)
                        adj[w].remove(u)
                        adj[v].remove(x)
                        adj[x].remove(v)
                        
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[w].add(x)
                        adj[x].add(w)
                    else:
                        current_girth = new_girth
                        swapped = True
                        break # Restart cycle finding with new graph
                
                if not swapped:
                    break # Stuck
    return None


def _find_short_cycle(adj, n, start):
    """Find a short cycle starting from start using BFS."""
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
            elif prev[v] != u and prev[u] != v:
                # Found a cycle
                # Reconstruct
                cycle = [v]
                curr = u
                while curr != v:
                    cycle.append(curr)
                    curr = prev[curr]
                cycle.append(v)
                return cycle
    return None


def _generate_random_regular(n, k, rng):
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
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


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    
    jobs = [
        ("circulant", lambda: _circulant_search(k, g)),
        ("local0", lambda: _local_search(k, g, seed=0)),
        ("local1", lambda: _local_search(k, g, seed=1)),
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
