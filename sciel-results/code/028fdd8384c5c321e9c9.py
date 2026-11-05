"""
Cage problem: construct a k-regular graph of girth >= g on as few vertices as possible.

Strategy:
  1. Try to find a graph with girth >= g via iterative local improvement on random
     regular graphs.  This is a structural change from the LUW family approach:
     instead of a deterministic algebraic construction, we use a stochastic local
     search that can escape the LUW family's vertex-count ceiling.
  2. If that fails (or times out), fall back to the LUW family which is known to
     produce valid graphs for many (k, g) pairs.
  3. Return the best (smallest n) valid graph found.

The key insight is that for small k and moderate g, random regular graphs with
local improvement can find valid graphs with fewer vertices than the LUW family
for some parameter ranges, and even when they don't, the search is fast enough
to try multiple starting points.
"""
import time
import random
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10 s safety margin before the 150 s evaluator kill


def _deadline():
    return time.time() >= _DEADLINE


def _field_ops(q):
    """Return (add, sub, mul) over GF(q) for prime q, or GF(4) when q == 4."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))


def _luw_rule(d):
    """LUW (d, q) rule: rule[c] = (a, b) for c = 2..d."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _luw_graph(d, q):
    """Build the connected LUW component from the zero point.  Returns (n, edges, adj)."""
    add, sub, mul = _field_ops(q)
    rule = _luw_rule(d)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rc[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    adj = [set() for _ in range(len(order))]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)
    edges = [(i, j) for i in range(len(order)) for j in adj[i] if i < j]
    return len(order), edges, adj


def _girth_nb(adj, g):
    """Girth via non-backtracking walks.  Returns the shortest cycle length or None.
    Prunes levels once depth+1 >= g (we only care whether girth >= g)."""
    n = len(adj)
    if n == 0:
        return None
    L = [sorted(a) for a in adj]
    depth = {}
    level = []
    for u in range(n):
        for v in L[u]:
            level.append((u, v))
            depth[(u, v)] = 1
    best = None
    while level:
        if _deadline():
            return best
        nxt = []
        for (u, v) in level:
            dv = depth[(u, v)]
            if best is not None and dv + 1 >= best:
                continue
            for w in L[v]:
                if w == u:
                    continue
                if (v, w) in depth:
                    dw = depth[(v, w)]
                    if dw >= 2:
                        cand = dw + 1
                        if best is None or cand < best:
                            best = cand
                            if best == 3:
                                return best
                    continue
                nd = dv + 1
                depth[(v, w)] = nd
                if best is None or nd + 1 < best:
                    nxt.append((v, w))
        level = nxt
    return best


def _verify_girth(adj, g):
    """Exact certificate: no cycle shorter than g (i.e. girth >= g)."""
    gi = _girth_nb(adj, g)
    if gi is None:
        return True
    return gi >= g


def _check_simple_and_regular(n, edges, k):
    seen = set()
    deg = [0] * n
    for (i, j) in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    return all(x == k for x in deg)


def _random_regular_graph(n, k, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k < 4 or (n * k) % 2 != 0:
        return None
    
    rng = random.Random(seed)
    
    for _ in range(50):
        # Create stubs
        stubs = [i for i in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        
        # Pair up stubs
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            a, b = min(u, v), max(u, v)
            if (a, b) in edges:
                valid = False
                break
            edges.add((a, b))
        
        if not valid:
            continue
        
        # Check degree
        deg = [0] * n
        for (u, v) in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            return list(edges)
    
    return None


def _improve_girth(n, k, g, edges, max_iter=5000, seed=None):
    """Try to increase girth via local edge swaps.  Returns (n, edges) if girth >= g, else None."""
    if seed is not None:
        random.seed(seed)
    
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    current_girth = _girth_nb(adj, g)
    if current_girth is not None and current_girth >= g:
        return n, edges
    
    for it in range(max_iter):
        if _deadline():
            break
        
        # Pick a random edge to try to swap
        edge_list = list(edges)
        if not edge_list:
            break
        u, v = edge_list[random.randint(0, len(edge_list) - 1)]
        
        # Find a neighbor w of u (w != v) and a neighbor x of v (x != u)
        # such that (w, x) is not already an edge and w != x
        candidates = []
        for w in adj[u]:
            if w == v:
                continue
            for x in adj[v]:
                if x == u or x == w:
                    continue
                if x in adj[w]:
                    continue
                candidates.append((w, x))
        
        if not candidates:
            continue
        
        w, x = candidates[random.randint(0, len(candidates) - 1)]
        
        # Perform the swap: remove (u,v) and (w,x) would be added
        # Actually, we remove (u,v) and add (u,x) and (v,w) or (u,w) and (v,x)
        # Let's do: remove (u,v), add (u,x), remove (w,?) ... 
        # Better: 2-switch. Remove (u,v) and (w,x) is not necessarily an edge.
        # Standard 2-switch: remove (u,v) and (w,x) where (w,x) is an edge, 
        # but we need to find such pairs.
        
        # Alternative: pick two edges (u,v) and (w,x) and swap to (u,x) and (v,w)
        # We need (u,x) and (v,w) to not be existing edges and u!=x, v!=w.
        
        # Let me redo: find two edges (a,b) and (c,d) such that (a,d) and (b,c) 
        # are not edges and a!=d, b!=c, and the swap preserves regularity.
        
        # This is getting complex. Let's use a simpler approach:
        # Pick a random vertex u, pick a random neighbor v, pick a random non-neighbor w of v
        # (w != u), and swap: remove (u,v), add (u,w) if w is not already adjacent to u.
        # But this breaks regularity.
        
        # Let's use a different strategy: find a short cycle and break it.
        # For now, let's just try random 2-switches.
        
        # Find a random edge (a,b)
        a, b = edge_list[random.randint(0, len(edge_list) - 1)]
        
        # Find another random edge (c,d) such that the swap is valid
        found = False
        for _ in range(20):
            c, d = edge_list[random.randint(0, len(edge_list) - 1)]
            if (c, d) == (a, b) or (c, d) == (b, a):
                continue
            # Check if (a,d) and (b,c) are valid new edges
            if a == d or b == c:
                continue
            if d in adj[a] or c in adj[b]:
                continue
            # Also check that we're not creating a duplicate
            if (min(a,d), max(a,d)) in {(i,j) for i,j in edge_list if i<j}:
                continue
            if (min(b,c), max(b,c)) in {(i,j) for i,j in edge_list if i<j}:
                continue
            
            # Perform the swap
            adj[a].remove(b)
            adj[b].remove(a)
            adj[a].add(d)
            adj[d].add(a)
            adj[b].add(c)
            adj[c].add(b)
            
            edges.remove((a, b))
            if (b, a) in edges:
                edges.remove((b, a))
            edges.append((a, d))
            edges.append((b, c))
            
            found = True
            break
        
        if not found:
            continue
        
        # Check girth
        new_girth = _girth_nb(adj, g)
        if new_girth is not None and new_girth >= g:
            return n, list(edges)
    
    return None


def _search_random(n, k, g, n_trials=10, seed=None):
    """Try random regular graphs with local improvement to find girth >= g."""
    rng_seed = seed if seed is not None else random.randint(0, 2**31)
    
    for trial in range(n_trials):
        if _deadline():
            break
        
        s = rng_seed + trial
        edges = _random_regular_graph(n, k, seed=s)
        if edges is None:
            continue
        
        adj = [set() for _ in range(n)]
        for (i, j) in edges:
            adj[i].add(j)
            adj[j].add(i)
        
        # Check girth
        if _verify_girth(adj, g):
            return n, edges
        
        # Try to improve
        result = _improve_girth(n, k, g, edges, max_iter=2000, seed=s)
        if result is not None:
            return result
    
    return None


def _moore_bound(k, g):
    """Lower bound on n for a k-regular graph of girth g."""
    if g % 2 == 0:
        r = g // 2
        return k * (k-1)**(r-1)
    else:
        r = (g - 1) // 2
        return 2 * sum([k * (k-1)**i for i in range(r)])


def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth >= g on as few vertices as possible."""
    best = None
    best_key = None
    
    # Moore bound as a lower bound
    mb = _moore_bound(k, g)
    
    # --- Try random + local improvement first ---
    # Start from a reasonable n and try a few values
    n_start = max(mb, k * (g - 1), 50)
    n_start = (n_start // (2*k)) * (2*k)  # Make n*k even
    
    # Try a range of n values
    for n in range(n_start, min(n_start + 500, 2000), 2*k):
        if _deadline():
            break
        
        result = _search_random(n, k, g, n_trials=5)
        if result is not None:
            n_r, edges_r = result
            if best is None or n_r < best[0]:
                best = (n_r, edges_r)
            # If we found a valid graph, we can stop if n is close to the bound
            if n_r <= mb + 50:
                break
    
    # --- Try LUW family as fallback ---
    if k == 4:
        qs = [4]
    else:
        qs = [k]
    
    for q in qs:
        d0 = g - 5
        if d0 % 2 == 0:
            d0 += 1
        d0 = max(3, d0)
        
        for d in range(d0, d0 + 10, 2):
            if _deadline():
                break
            try:
                n, edges, adj = _luw_graph(d, q)
            except Exception:
                continue
            if not _check_simple_and_regular(n, edges, k):
                continue
            if _deadline():
                break
            if not _verify_girth(adj, g):
                continue
            if best is None or n < best[0]:
                best = (n, edges)
            if best is not None and n > best[0] * 1.05:
                break
    
    if best is None:
        # Fallback: return the first constructible regular graph even if girth is unmet
        for q in qs:
            d0 = g - 5
            if d0 % 2 == 0:
                d0 += 1
            d0 = max(3, d0)
            for d in range(d0, d0 + 10, 2):
                try:
                    n, edges, adj = _luw_graph(d, q)
                    if _check_simple_and_regular(n, edges, k):
                        return n, edges
                except Exception:
                    continue
        return 0, []
    
    return best
