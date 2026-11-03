import time
import threading
from collections import deque
import random
import sys
import numpy as np

_DEADLINE = 145.0
_start = [None]

def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s

def _nonback_girth_fast(adj, n, target_girth):
    """
    Optimized non-backtracking girth check.
    Returns the shortest cycle length found, or 10**9 if none shorter than target_girth.
    """
    prev = {}
    q = deque()
    max_dist = target_girth - 1
    
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            if key not in prev:
                prev[key] = (u, 1)
                q.append(key)
    
    while q:
        (v, p), (u, dist) = q.popleft()
        if dist >= max_dist:
            continue
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
    return 10**9

def _is_valid(adj, n, k, g):
    if n == 0: return False
    if any(len(adj[i]) != k for i in range(n)): return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]): return False
        for j in adj[i]:
            if i <= j: return False
    return _nonback_girth_fast(adj, n, g) >= g

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj

def _local_moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        return 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        return 1 + k * sum((k-1)**i for i in range(r))

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

def _circulant_search(k, g):
    if k != 5: return None
    best_n = 10**9
    best_edges = None
    mb = _local_moore_bound(k, g)
    if mb % 2 != 0:
        mb += 1
    
    # For 5-regular circulant on Z_n, we need n even and S = {a, -a, b, -b, n/2}
    # with a, b, n/2 all distinct and nonzero mod n
    # Quick girth check via BFS
    for n in range(mb, mb + 3000, 2):
        if _elapsed() > _DEADLINE - 15.0: break
        half_n = n // 2
        for a in range(1, half_n):
            if _elapsed() > _DEADLINE - 15.0: break
            for b in range(a + 1, half_n):
                if _elapsed() > _DEADLINE - 15.0: break
                S = [a, n-a, b, n-b, half_n]
                s_set = set(S)
                if len(s_set) != 5: continue
                
                # Quick 3-cycle check
                has_3 = False
                for i in range(5):
                    for j in range(i+1, 5):
                        if (S[i] + S[j]) % n in s_set:
                            has_3 = True
                            break
                    if has_3: break
                if has_3: continue
                
                # Quick 4-cycle check
                sums = set()
                has_4 = False
                for i in range(5):
                    for j in range(i+1, 5):
                        s_val = (S[i] + S[j]) % n
                        if s_val in sums:
                            has_4 = True
                            break
                        sums.add(s_val)
                    if has_4: break
                if has_4: continue
                
                # BFS for girth
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
                    if dist >= g: break
                    for s in S:
                        if s == neg[last_s]: continue
                        new_v = (v + s) % n
                        if new_v == 0:
                            girth_found = dist + 1
                            break
                        new_state = (new_v, s)
                        if new_state not in visited:
                            visited.add(new_state)
                            q.append((new_v, s, dist + 1))
                    if girth_found <= g: break
                
                if girth_found >= g:
                    if n < best_n:
                        best_n = n
                        adj = [set() for _ in range(n)]
                        for i in range(n):
                            for s in S:
                                adj[i].add((i + s) % n)
                        if all(len(adj[i]) == 5 for i in range(n)):
                            if _nonback_girth_fast(adj, n, g) >= g:
                                best_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                                return best_n, best_edges
    
    if best_edges:
        return best_n, best_edges
    return None

def _anneal_search(k, g, seed=0):
    rng = random.Random(seed)
    mb = _local_moore_bound(k, g)
    if mb % 2 != 0:
        mb += 1
    
    # Try to build a random k-regular graph and improve girth via 2-switches
    for n in [mb, mb + 20, mb + 50, mb + 100, mb + 200]:
        if _elapsed() > _DEADLINE - 5.0: break
        
        # Generate random k-regular graph using configuration model
        adj = [set() for _ in range(n)]
        success = False
        for attempt in range(20):
            adj = [set() for _ in range(n)]
            stubs = [i for i in range(n) for _ in range(k)]
            rng.shuffle(stubs)
            valid = True
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid and all(len(adj[i]) == k for i in range(n)):
                success = True
                break
        
        if not success:
            continue
        
        # Check if already valid
        if _nonback_girth_fast(adj, n, g) >= g:
            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            return n, edges
        
        # 2-switch operations to increase girth
        # A 2-switch removes edges (u,v) and (u',v') and adds (u,v') and (u',v)
        # This preserves k-regularity
        for _ in range(20000):
            if _elapsed() > _DEADLINE - 5.0: break
            
            # Pick two random edges
            u = rng.randint(0, n-1)
            v = rng.choice(list(adj[u]))
            u2 = rng.randint(0, n-1)
            v2 = rng.choice(list(adj[u2]))
            
            # Ensure we have distinct vertices
            if len({u, v, u2, v2}) < 4:
                continue
            # Ensure (u,v) and (u2,v2) are different edges
            if (u, v) == (u2, v2) or (u, v) == (v2, u2):
                continue
            
            # Remove edges (u,v) and (u2,v2)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            # Add edges (u,v2) and (u2,v) if they don't exist
            if v2 not in adj[u] and v not in adj[u2]:
                adj[u].add(v2)
                adj[v2].add(u)
                adj[u2].add(v)
                adj[v].add(u2)
            else:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                adj[u2].add(v2)
                adj[v2].add(u2)
        
        # Final check
        if _nonback_girth_fast(adj, n, g) >= g:
            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            return n, edges
            
    return None

def _worker(name, fn, holder):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None: return
    n, edges = res
    if _elapsed() >= _DEADLINE: return
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
    
    # Run LUW first (fast, known good construction)
    try:
        n_luw, edges_luw = _luw(k, g)
        if n_luw > 0 and _is_valid(_build_adj(edges_luw, n_luw), n_luw, k, g):
            holder["best"] = (n_luw, edges_luw)
    except:
        pass
    
    # Run circulant and annealing in parallel
    jobs = [
        ("circ", lambda: _circulant_search(k, g)),
        ("anneal0", lambda: _anneal_search(k, g, 0)),
        ("anneal1", lambda: _anneal_search(k, g, 1)),
        ("anneal2", lambda: _anneal_search(k, g, 2)),
    ]
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder), daemon=True)
        t.start()
        threads.append(t)
        
    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads): break
        time.sleep(0.05)
        
    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
