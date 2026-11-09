import time
import sys
import numpy as np
from collections import deque, defaultdict
import random
import math

_DEADLINE = 140.0
_T0 = None

def _elapsed():
    return time.time() - _T0

def _time_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin

def construct(k, g):
    global _T0
    _T0 = time.time()
    
    # Handle small cases with hardcoded cages
    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return (n, edges)
    
    if k == 3:
        if g <= 4:
            return (4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
        if g <= 5:
            # Petersen
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((5 + i, 5 + (i + 2) % 5))
                edges.append((i, 5 + i))
            return (10, edges)
        if g <= 6:
            # Heawood
            edges = []
            fano_lines = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
            for li, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7 + li))
            return (14, edges)
        if g <= 7:
            # Petersen-like or use LUW
            pass
        # For k=3, g>=7, try LUW with q=3
        result = _luw_construct(k, g)
        if result is not None:
            return result
        # Fallback: complete bipartite K_{3,3} has girth 4, not good enough
        # Use a known (3,7)-cage: the Petersen graph doesn't work for g=7
        # Actually for (3,7) the smallest is 24 vertices (Heawood is (3,6))
        # Let's try to construct or return a regular graph
        return _simple_regular_graph(k)
    
    # For k >= 4, try LUW construction
    result = _luw_construct(k, g)
    if result is not None:
        return result
    
    # Try circulant graph construction
    result = _circulant_girth_construct(k, g)
    if result is not None:
        return result
    
    # Try random greedy construction
    result = _greedy_girth_construct(k, g)
    if result is not None:
        return result
    
    # Fallback: return a simple regular graph
    return _simple_regular_graph(k)

def _luw_construct(k, g, d_override=None):
    """Build Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=k."""
    q = k
    if q not in (2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        return None

    if d_override is not None:
        d = d_override
    else:
        d = None
        for cand in range(3, 25):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

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

    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q

    def encode(side, x):
        val = 0
        for c in range(1, d + 1):
            val += x[c] * q_powers[d - c]
        return side * (q ** d) + val

    def decode(enc):
        side = enc // (q ** d)
        val = enc % (q ** d)
        x = [0] * (d + 1)
        for c in range(1, d + 1):
            x[c] = val % q
            val //= q
        return side, x

    def neighbours(enc):
        side, x = decode(enc)
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield encode(1 - side, y)

    start_enc = encode(0, [0] * (d + 1))
    index = {start_enc: 0}
    order = [start_enc]
    edges = []

    for i, enc in enumerate(order):
        if not _time_ok(margin=15.0):
            return None
        side, _ = decode(enc)
        for other_enc in neighbours(enc):
            if other_enc not in index:
                index[other_enc] = len(order)
                order.append(other_enc)
            if side == 0:
                edges.append((index[enc], index[other_enc]))

    n = len(order)
    if n == 0:
        return None

    if _verify(n, edges, k, g):
        return (n, edges)
    return None

def _circulant_girth_construct(k, g):
    """Try to construct a circulant graph with girth >= g."""
    # For a circulant graph on Z_n with jumps S, the graph is k-regular
    # if |S| = k and S is symmetric (if s in S then -s in S, or s = -s)
    # For odd k, we need one jump to be self-inverse (s = n/2, so n must be even)
    # For even k, we need k/2 pairs of jumps
    
    if k % 2 == 1:
        # Need n even, and one jump = n//2
        # So we need (k-1)/2 other jumps
        half_k = (k - 1) // 2
    else:
        half_k = k // 2
    
    # Try various n values
    # Moore bound: n >= 2 * sum_{i=0}^{floor((g-1)/2)} (k-1)^i
    moore = 2
    for i in range(1, (g - 1) // 2 + 1):
        moore += (k - 1) ** i
    moore = max(moore, k + 1)
    
    # Try n from Moore bound upward
    max_n = moore + 500
    
    for n in range(moore, max_n + 1):
        if not _time_ok(margin=10.0):
            break
        
        if k % 2 == 1 and n % 2 != 0:
            continue
        
        # Try random sets of jumps
        for attempt in range(50):
            if not _time_ok(margin=5.0):
                break
            
            if k % 2 == 1:
                # One jump is n//2, need half_k more
                jumps = [n // 2]
                candidates = list(range(1, n // 2))
                random.shuffle(candidates)
                for s in candidates:
                    if len(jumps) >= half_k + 1:
                        break
                    # Check if s and -s mod n are both not already in jumps
                    neg_s = (-s) % n
                    if s not in jumps and neg_s not in jumps:
                        jumps.append(s)
            else:
                # Need half_k pairs
                jumps = []
                candidates = list(range(1, n))
                random.shuffle(candidates)
                for s in candidates:
                    if len(jumps) >= half_k * 2:
                        break
                    neg_s = (-s) % n
                    if s not in jumps and neg_s not in jumps:
                        jumps.append(s)
                        jumps.append(neg_s)
            
            if len(jumps) < k:
                continue
            
            # Check girth
            if _check_circulant_girth(n, jumps, g):
                # Build edges
                edges = []
                for i in range(n):
                    for s in jumps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return (n, edges)
    
    return None

def _check_circulant_girth(n, jumps, g):
    """Check if circulant graph with given jumps has girth >= g."""
    visited = set()
    queue = deque()
    queue.append((0, None, 0))
    visited.add((0, None))
    
    while queue:
        v, prev_jump, depth = queue.popleft()
        if depth >= g - 1:
            continue
        
        for s in jumps:
            if prev_jump is not None and s == (-prev_jump) % n:
                continue
            
            new_v = (v + s) % n
            new_state = (new_v, s)
            
            if new_v == 0 and depth + 1 < g:
                return False
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_v, s, depth + 1))
    
    return True

def _greedy_girth_construct(k, g, max_vertices=5000, max_attempts=20):
    """Random greedy algorithm to construct a k-regular graph with girth >= g."""
    for attempt in range(max_attempts):
        if not _time_ok(margin=10.0):
            break
        
        # Start with a few vertices
        n = 0
        adj = []  # adj[v] = list of neighbors
        degrees = []  # degrees[v] = current degree
        
        # Add initial vertices
        for _ in range(k):
            n += 1
            adj.append([])
            degrees.append(0)
        
        # Try to build up the graph
        for step in range(max_vertices - k):
            if not _time_ok(margin=5.0):
                break
            
            # Find vertices with degree < k
            low_degree = [v for v in range(n) if degrees[v] < k]
            if not low_degree:
                break
            
            # Pick a random low-degree vertex
            v = random.choice(low_degree)
            remaining = k - degrees[v]
            
            # Try to connect v to other low-degree vertices
            candidates = [u for u in range(n) if u != v and degrees[u] < k]
            random.shuffle(candidates)
            
            connected = 0
            for u in candidates:
                if connected >= remaining:
                    break
                if u == v:
                    continue
                
                # Check if adding edge (v, u) would create a cycle shorter than g
                if _would_create_short_cycle(adj, v, u, g):
                    continue
                
                # Add the edge
                adj[v].append(u)
                adj[u].append(v)
                degrees[v] += 1
                degrees[u] += 1
                connected += 1
            
            # If we couldn't connect enough, try adding a new vertex
            if connected < remaining:
                # Add a new vertex and connect to v
                n += 1
                adj.append([])
                degrees.append(0)
                
                # Connect new vertex to v
                adj[v].append(n - 1)
                adj[n - 1].append(v)
                degrees[v] += 1
                degrees[n - 1] += 1
                connected += 1
                
                # Connect new vertex to other low-degree vertices
                remaining2 = k - degrees[n - 1]
                candidates2 = [u for u in range(n) if u != n - 1 and degrees[u] < k]
                random.shuffle(candidates2)
                
                for u in candidates2:
                    if connected >= remaining:
                        break
                    if _would_create_short_cycle(adj, n - 1, u, g):
                        continue
                    
                    adj[n - 1].append(u)
                    adj[u].append(n - 1)
                    degrees[n - 1] += 1
                    degrees[u] += 1
                    connected += 1
        
        # Check if all vertices have degree k
        if all(deg == k for deg in degrees):
            # Build edge list
            edges = []
            for v in range(n):
                for u in adj[v]:
                    if v < u:
                        edges.append((v, u))
            
            # Verify the graph
            if _verify(n, edges, k, g):
                return (n, edges)
    
    return None

def _would_create_short_cycle(adj, v, u, g):
    """Check if adding edge (v, u) would create a cycle shorter than g."""
    # BFS from v to u, avoiding the direct edge
    # We want to find if there's a path from v to u of length < g-1
    # that doesn't use the edge (v, u)
    
    # If v and u are already adjacent, adding another edge would be a multigraph
    if u in adj[v]:
        return True
    
    # BFS from v
    visited = {v: 0}
    queue = deque([(v, 0)])
    
    while queue:
        node, depth = queue.popleft()
        if depth >= g - 2:  # We're looking for paths of length < g-1
            continue
        
        for neighbor in adj[node]:
            if neighbor == u:
                # Found a path from v to u of length depth + 1
                # Total cycle length would be depth + 1 + 1 = depth + 2
                if depth + 2 < g:
                    return True
                continue
            
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
    
    return False

def _verify(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g."""
    if n == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    for start in range(n):
        if not _time_ok(margin=10.0):
            return False
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                new_state = (w, v)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if w == start and depth + 1 < g:
                    return False
                queue.append((w, v, depth + 1))
    
    return True

def _simple_regular_graph(k):
    """Fallback: return a simple k-regular graph."""
    n = k + 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return (n, edges)
