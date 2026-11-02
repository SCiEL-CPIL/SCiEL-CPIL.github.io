"""Constructive search for small k-regular graphs of large girth.

Strategy:
  1. Estimate candidate vertex count n from Moore bound upward.
  2. Generate random k-regular simple graphs using improved configuration model.
  3. Find and break short cycles via targeted 2-switches.
  4. Certify girth with non-backtracking BFS.
  5. Return first valid graph or fall back to algebraic construction.
"""
import random
import time
import math
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 110.0


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _moore_bound(k, g):
    """Moore bound (lower bound) for n(k, g)."""
    if g % 2 == 0:
        h = g // 2
        if k == 2:
            return 2 * h
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s
    else:
        h = (g - 1) // 2
        if k == 2:
            return g
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s


def _generate_regular(n, k, rng):
    """Generate a random k-regular simple graph using improved method."""
    if n * k % 2 != 0 or k >= n or n < 3:
        return None
    
    # Use configuration model with retry
    for attempt in range(50):
        adj = [set() for _ in range(n)]
        # Create stubs
        stubs = []
        for v in range(n):
            stubs.extend([v] * k)
        rng.shuffle(stubs)
        
        success = True
        i = 0
        while i < len(stubs) - 1:
            a, b = stubs[i], stubs[i + 1]
            i += 2
            if a == b or b in adj[a]:
                # Try to find alternative pairing
                fixed = False
                for j in range(i, len(stubs), 2):
                    c, d = stubs[j], stubs[j + 1]
                    if c == d:
                        continue
                    # Check if swapping helps
                    if a not in adj[c] and b not in adj[d] and c != b and d != a:
                        if c not in adj[a] and d not in adj[b]:
                            # Swap b with c, d with a
                            adj[a].add(c)
                            adj[c].add(a)
                            adj[b].add(d)
                            adj[d].add(b)
                            stubs[i+1], stubs[j], stubs[j+1], stubs[i] = c, b, d, a
                            fixed = True
                            break
                if not fixed:
                    success = False
                    break
            else:
                adj[a].add(b)
                adj[b].add(a)
        
        if success:
            # Verify regularity and simplicity
            valid = True
            for v in range(n):
                if len(adj[v]) != k:
                    valid = False
                    break
            if valid:
                for v in range(n):
                    for u in adj[v]:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                return adj
    
    return None


def _find_short_cycle(adj, n, g, max_search=5000):
    """Find a cycle of length < g using BFS. Returns (cycle_vertices, length) or (None, None)."""
    # Sample random start vertices
    starts = random.sample(range(n), min(n, 50))
    
    for s in starts:
        # Non-backtracking BFS: state is (prev, cur, dist)
        visited = {}
        parent = {}
        dq = deque()
        
        for u in adj[s]:
            dq.append((s, u, 1))
            visited[(s, u)] = (s, u, 1)
            parent[(s, u)] = None
        
        while dq and len(visited) < max_search:
            prev, cur, dist = dq.popleft()
            if dist >= g:
                continue
            
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s:
                    # Found cycle of length dist+1
                    if dist + 1 < g:
                        # Reconstruct cycle
                        cycle = [cur, s]
                        state = (prev, cur)
                        while state is not None:
                            p, c = state
                            cycle.append(p)
                            state = parent.get(state)
                        cycle.reverse()
                        return cycle, dist + 1
                    continue
                
                if (cur, nxt) in visited:
                    old_dist = visited[(cur, nxt)][2]
                    cyc_len = dist + 1 - old_dist
                    if cyc_len < g:
                        # Reconstruct
                        cycle = [cur, nxt]
                        state = (prev, cur)
                        while state is not None:
                            p, c = state
                            cycle.append(p)
                            state = parent.get(state)
                        cycle.reverse()
                        return cycle, cyc_len
                    continue
                
                visited[(cur, nxt)] = (cur, nxt, dist + 1)
                parent[(cur, nxt)] = (prev, cur)
                dq.append((cur, nxt, dist + 1))
    
    return None, None


def _try_2switch_on_cycle(adj, n, k, cycle, rng):
    """Try to break a cycle using a 2-switch. Returns True if successful."""
    if len(cycle) < 3:
        return False
    
    # Try different 2-switches on edges of the cycle
    m = len(cycle)
    for _ in range(100):
        # Pick two non-adjacent edges on the cycle
        i = rng.randrange(m)
        j = (i + rng.randint(2, m - 2)) % m
        if j == i or j == (i + 1) % m or j == (i - 1) % m:
            continue
        
        # Edges: (cycle[i], cycle[(i+1)%m]) and (cycle[j], cycle[(j+1)%m])
        a, b = cycle[i], cycle[(i + 1) % m]
        c, d = cycle[j], cycle[(j + 1) % m]
        
        # Check simplicity: new edges (a,c) and (b,d) or (a,d) and (b,c)
        for swap_type in range(2):
            if swap_type == 0:
                new1, new2 = (a, c), (b, d)
            else:
                new1, new2 = (a, d), (b, c)
            
            x1, y1 = new1
            x2, y2 = new2
            
            # Check all constraints
            if x1 == y1 or x2 == y2:
                continue
            if y1 in adj[x1] or y2 in adj[x2]:
                continue
            if x1 in adj[y2] or x2 in adj[y1]:
                continue
            if x1 in adj[x2] or y1 in adj[y2]:
                continue
            
            # Perform the 2-switch
            adj[a].discard(b)
            adj[b].discard(a)
            adj[c].discard(d)
            adj[d].discard(c)
            
            adj[x1].add(y1)
            adj[y1].add(x1)
            adj[x2].add(y2)
            adj[y2].add(x2)
            
            return True
    
    return False


def _girth_certify(adj, n, g):
    """Full girth certification. Returns (is_valid, min_girth)."""
    min_girth = g
    
    for s in range(n):
        visited = {}
        dq = deque()
        
        for u in adj[s]:
            dq.append((s, u, 1))
            visited[(s, u)] = 1
        
        while dq:
            prev, cur, dist = dq.popleft()
            if dist >= g:
                continue
            
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s:
                    if dist + 1 < min_girth:
                        min_girth = dist + 1
                    continue
                if (cur, nxt) in visited:
                    if dist + 1 < visited[(cur, nxt)]:
                        cyc = dist + 1 - visited[(cur, nxt)]
                        if cyc < min_girth:
                            min_girth = cyc
                    continue
                visited[(cur, nxt)] = dist + 1
                dq.append((cur, nxt, dist + 1))
        
        if min_girth < g:
            break
    
    return (min_girth >= g), min_girth


def _try_build(n, k, g, rng):
    """Try to build a k-regular graph on n vertices with girth >= g."""
    max_attempts = max(10, int(30 * _time_left() / 30.0))
    
    for attempt in range(max_attempts):
        if _time_left() < 3.0:
            return None
        
        adj = _generate_regular(n, k, rng)
        if adj is None:
            continue
        
        # Quick check from subset
        cycle, clen = _find_short_cycle(adj, n, g, max_search=2000)
        if cycle is None:
            ok, mg = _girth_certify(adj, n, g)
            if ok:
                return adj
        
        # Repair loop
        for repair_iter in range(30):
            if _time_left() < 3.0:
                break
            
            cycle, clen = _find_short_cycle(adj, n, g, max_search=3000)
            if cycle is None:
                ok, mg = _girth_certify(adj, n, g)
                if ok:
                    return adj
                break
            
            # Try to break this cycle
            if not _try_2switch_on_cycle(adj, n, k, cycle, rng):
                # Try random 2-switch
                for _ in range(20):
                    v = rng.randrange(n)
                    if len(adj[v]) < 2:
                        continue
                    u = rng.choice(list(adj[v]))
                    x = rng.randrange(n)
                    if x in (v, u):
                        continue
                    if len(adj[x]) < 1:
                        continue
                    y = rng.choice(list(adj[x]))
                    if y in (v, u) or x == y:
                        continue
                    if x in adj[v] or y in adj[u]:
                        continue
                    if v in adj[y] or u in adj[x]:
                        continue
                    
                    adj[v].discard(u)
                    adj[u].discard(v)
                    adj[x].discard(y)
                    adj[y].discard(x)
                    adj[v].add(x)
                    adj[x].add(v)
                    adj[u].add(y)
                    adj[y].add(u)
                    break
    
    return None


def _fallback_luw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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


def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)

    if k < 2 or g < 3:
        return _fallback_luw(max(2, k), max(3, g))

    moore = _moore_bound(k, g)
    
    # Candidate sizes: start near Moore bound, grow gradually
    max_n = min(max(moore, 2 * k * k), 2500)
    
    candidates = []
    n_try = max(moore, 2 * k)
    while n_try <= max_n:
        if n_try * k % 2 == 0:
            candidates.append(n_try)
        n_try = int(n_try * 1.2) + 1
    
    if not candidates:
        n_try = max(moore, 2 * k)
        if n_try * k % 2 != 0:
            n_try += 1
        candidates = [n_try]
    
    best_n = None
    best_adj = None
    
    for n in candidates:
        if _time_left() < 5.0:
            break
        
        adj = _try_build(n, k, g, rng)
        if adj is not None:
            edges = []
            for v in range(n):
                for u in adj[v]:
                    if v < u:
                        edges.append((v, u))
            return n, edges
    
    return _fallback_luw(k, g)
