"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise. Optimized with early termination."""
    for start in range(n):
        # State: (vertex, prev_vertex) -> depth
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(q, g):
    """LUW (Lubotzky-Phillips-Sarnak / Marcus-Spielman-Srivastava style) construction.
    Builds a (q+1)-regular graph of girth at least g using voltage graph over Z_q.
    Returns (n, edges)."""
    if q == 3:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    elif q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    else:
        # Generic prime q: use the standard LPS construction
        p = q
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%p), (lambda x,y:(x-y)%p), (lambda x,y:x*y%p)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(p):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

def _build_circulant(n, shifts):
    """Build a circulant graph C(n, shifts) where shifts is a list of offsets.
    For 3-regular: shifts must have 3 elements (or 2 if one is n/2)."""
    adj = [set() for _ in range(n)]
    edges = []
    for s in shifts:
        s = s % n
        for i in range(n):
            j = (i + s) % n
            if j not in adj[i]:
                adj[i].add(j)
                adj[j].add(i)
                if i < j:
                    edges.append((i, j))
    # Check all vertices have degree 3
    for i in range(n):
        if len(adj[i]) != 3:
            return None, None
    # Check connectivity
    visited = set([0])
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None, None
    return n, edges

def _circulant_girth_check(n, shifts, g):
    """Quick girth check for circulant graph using BFS from vertex 0.
    Since circulant graphs are vertex-transitive, checking from 0 is sufficient
    for finding the shortest cycle through 0, but we need to check all vertices
    for the true girth. However, for a quick filter, we check from 0."""
    adj = [set() for _ in range(n)]
    for s in shifts:
        s = s % n
        for i in range(n):
            j = (i + s) % n
            adj[i].add(j)
            adj[j].add(i)
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    # Full girth check
    return _verify_girth_fast(adj, n, g)

def _search_circulant(n, g, seed, max_shifts=None):
    """Search for a 3-regular circulant graph with girth >= g.
    For 3-regular circulant: C(n, {a, b, c}) where a+b+c=0 mod n or one shift is n/2."""
    rng = _random.Random(seed)
    
    if n % 2 == 0:
        # One shift can be n/2, so we need 2 more shifts
        # C(n, {n/2, a, b}) where a != b, a != n-a, etc.
        # Actually for 3-regular: shifts are {s1, s2, s3} with no redundancy
        # If n is even, n/2 is self-inverse, so we can use {n/2, a, -a} but that gives degree 3 only if a != n/2
        # Let's use {a, b, c} where all are distinct and none is 0 or n/2 (or one is n/2)
        
        # Option 1: {n/2, a, b} where a != b, a != -b, neither is n/2
        valid_shifts_a = [s for s in range(1, n//2) if s != n//2]
        if len(valid_shifts_a) < 2:
            return None
        
        for attempt in range(max_shifts or 200):
            if _time.time() > _circulant_deadline:
                break
            a = rng.choice(valid_shifts_a)
            b = rng.choice(valid_shifts_a)
            if a == b or a == n - b:
                continue
            shifts = [n//2, a, b]
            # Check no duplicate edges
            adj_check = set()
            valid = True
            for s in shifts:
                for i in range(n):
                    j = (i + s) % n
                    key = (min(i,j), max(i,j))
                    if key in adj_check:
                        valid = False
                        break
                    adj_check.add(key)
                if not valid:
                    break
            if not valid:
                continue
            if _circulant_girth_check(n, shifts, g):
                edges = []
                for s in shifts:
                    for i in range(n):
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    else:
        # n is odd: shifts {a, b, c} with a+b+c != 0 generally
        # For 3-regular: {a, b, c} where all distinct, none is 0
        valid_shifts = list(range(1, n))
        for attempt in range(max_shifts or 200):
            if _time.time() > _circulant_deadline:
                break
            shifts = rng.sample(valid_shifts, 3)
            # Check no duplicate edges (i.e., no s1 == s2, s1 == -s2, etc.)
            adj_check = set()
            valid = True
            for s in shifts:
                for i in range(n):
                    j = (i + s) % n
                    key = (min(i,j), max(i,j))
                    if key in adj_check:
                        valid = False
                        break
                    adj_check.add(key)
                if not valid:
                    break
            if not valid:
                continue
            if _circulant_girth_check(n, shifts, g):
                edges = []
                for s in shifts:
                    for i in range(n):
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    return None

def _local_swap_optimize(n, edges, g, seed, max_swaps=5000, time_limit=30):
    """Try to improve girth via edge swaps. Returns (n, edges) if improved, else None."""
    rng = _random.Random(seed)
    start_time = _time.time()
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Current girth check
    if not _verify_girth_fast(adj, n, g):
        return None
    
    # Try edge swaps: pick two non-adjacent edges (a,b) and (c,d) where a!=c, etc.
    # and swap to (a,c) and (b,d) or (a,d) and (b,c)
    # This preserves 3-regularity if we're careful
    
    # Represent edges as a list
    edge_list = list(edges)
    
    for swap_iter in range(max_swaps):
        if _time.time() - start_time > time_limit:
            break
        
        # Pick two random edges
        i1 = rng.randint(0, len(edge_list) - 1)
        i2 = rng.randint(0, len(edge_list) - 1)
        if i1 == i2:
            continue
        
        a, b = edge_list[i1]
        c, d = edge_list[i2]
        
        # Check they don't share vertices
        if a == c or a == d or b == c or b == d:
            continue
        
        # Try swap: remove (a,b) and (c,d), add (a,c) and (b,d)
        new_edges = edge_list[:i1] + edge_list[i1+1:i2] + edge_list[i2+1:]
        if i1 > i2:
            new_edges = edge_list[:i2] + edge_list[i2+1:i1] + edge_list[i1+1:]
        
        # Check new edges don't already exist
        if c in adj[a] or d in adj[b]:
            continue
        # Also check (a,d) and (b,c) don't exist for the other swap option
        # Let's try (a,c) and (b,d) first
        if c in adj[a] or d in adj[b]:
            # Try (a,d) and (b,c)
            if d in adj[a] or c in adj[b]:
                continue
            new_edges.append((a, d))
            new_edges.append((b, c))
        else:
            new_edges.append((a, c))
            new_edges.append((b, d))
        
        # Build new adjacency and check girth
        new_adj = [set() for _ in range(n)]
        for u, v in new_edges:
            if u == v:
                break
            new_adj[u].add(v)
            new_adj[v].add(u)
        else:
            # Check degrees
            valid = True
            for i in range(n):
                if len(new_adj[i]) != 3:
                    valid = False
                    break
            if valid and _verify_girth_fast(new_adj, n, g):
                # Check connectivity
                visited = set([0])
                stack = [0]
                while stack:
                    v = stack.pop()
                    for w in new_adj[v]:
                        if w not in visited:
                            visited.add(w)
                            stack.append(w)
                if len(visited) == n:
                    edge_list = new_edges
                    adj = new_adj
                    continue
            # Revert
            pass
    
    return n, edge_list

def construct(k, g):
    global _circulant_deadline
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        # Use LUW for other values of k
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: LUW construction as baseline (fast and reliable)
    n_luw, edges_luw = _luw_construct(3, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # Strategy 2: Search for circulant graphs with n < best_n
    # Circulant graphs C(n, {a,b,c}) can sometimes achieve good girth
    # Start from moore bound and go up to best_n
    _circulant_deadline = deadline - 10
    
    # For circulant 3-regular graphs, n must be such that we can find valid shifts
    # Try various n values
    n_start = max(moore, 20)
    if n_start % 2 != 0:
        n_start += 1
    
    circulant_found = False
    for n_try in range(n_start, min(best_n, n_start + 2000), 2):
        if _time.time() > _circulant_deadline:
            break
        
        # Try circulant with different seeds
        for seed in range(5):
            if _time.time() > _circulant_deadline:
                break
            result = _search_circulant(n_try, g, seed=seed * 1000003 + n_try, max_shifts=100)
            if result is not None:
                n_circ, edges_circ = result
                if n_circ < best_n:
                    best_n = n_circ
                    best_result = (n_circ, edges_circ)
                    circulant_found = True
                break
    
    # Strategy 3: Local swap optimization on LUW result to see if we can reduce
    # This is unlikely to reduce n, but might find alternative structures
    # Actually, swaps don't change n, so this won't help with n minimization
    # Skip this for now
    
    # Strategy 4: Try more circulant searches with more attempts if time allows
    if _time.time() < deadline - 15 and not circulant_found:
        for n_try in range(n_start, min(best_n, n_start + 500), 2):
            if _time.time() > deadline - 15:
                break
            for seed in range(20):
                if _time.time() > deadline - 15:
                    break
                result = _search_circulant(n_try, g, seed=seed * 7919 + n_try * 13, max_shifts=50)
                if result is not None:
                    n_circ, edges_circ = result
                    if n_circ < best_n:
                        best_n = n_circ
                        best_result = (n_circ, edges_circ)
                        circulant_found = True
                    break
    
    return best_result
# EVOLVE-BLOCK-END
