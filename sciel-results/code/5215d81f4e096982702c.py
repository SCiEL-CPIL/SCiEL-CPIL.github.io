import time
import random
import sys
from collections import deque
import numpy as np

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if depth >= g - 1:
                continue
                
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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
            yield 1 - side, tuple(y)

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
                
    return len(order), edges

def _count_short_cycles(n, edges, g):
    """Count cycles of length < g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    count = 0
    for s in range(n):
        visited = {s: 0}
        dq = deque()
        for nb in adj[s]:
            dq.append((nb, s, 1))
            visited[(nb, s)] = 1
        
        while dq:
            curr, prev, depth = dq.popleft()
            if depth >= g - 1:
                continue
            if curr == s:
                if depth + 1 < g:
                    count += 1
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in visited:
                    visited[state] = depth + 1
                    dq.append((nb, curr, depth + 1))
    
    return count

def _random_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices.
    """
    rng = random.Random(seed)
    
    for attempt in range(200):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        rng.shuffle(stubs)
        
        edges = set()
        valid = True
        adj_count = [0] * n
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
            adj_count[u] += 1
            adj_count[v] += 1
            if adj_count[u] > 4 or adj_count[v] > 4:
                valid = False
                break
        
        if valid and all(c == 4 for c in adj_count):
            return list(edges)
            
    return None

def _improve_girth(n, edges, g, time_limit, rng):
    """
    Local search to eliminate short cycles.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    best_edges = list(edges)
    best_count = _count_short_cycles(n, edges, g)
    
    if best_count == 0:
        return edges, 0
    
    t_start = time.time()
    temp = 1.0
    cooling = 0.9995
    
    while time.time() < t_start + time_limit:
        if best_count == 0:
            break
        
        # Pick a random edge
        e1 = rng.choice(edges)
        u1, v1 = e1
        
        # Try to swap with another edge
        e2 = rng.choice(edges)
        u2, v2 = e2
        
        if e1 == e2:
            continue
        
        # Check if swap is valid
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
        
        new_e1 = (min(u1, u2), max(u1, u2))
        new_e2 = (min(v1, v2), max(v1, v2))
        
        if new_e1 in edge_set or new_e2 in edge_set:
            continue
        
        # Perform swap
        adj[u1].discard(v1)
        adj[v1].discard(u1)
        adj[u2].discard(v2)
        adj[v2].discard(u2)
        
        adj[u1].add(u2)
        adj[u2].add(u1)
        adj[v1].add(v2)
        adj[v2].add(v1)
        
        edge_set.discard(e1)
        edge_set.discard(e2)
        edge_set.add(new_e1)
        edge_set.add(new_e2)
        
        edges.remove(e1)
        edges.remove(e2)
        edges.append(new_e1)
        edges.append(new_e2)
        
        new_count = _count_short_cycles(n, edges, g)
        
        if new_count < best_count:
            best_count = new_count
            best_edges = list(edges)
        
        if new_count <= best_count or rng.random() < pow(2.718, -(new_count - best_count) / temp):
            pass  # accept
        else:
            # revert
            adj[u1].discard(u2)
            adj[u2].discard(u1)
            adj[v1].discard(v2)
            adj[v2].discard(v1)
            
            adj[u1].add(v1)
            adj[v1].add(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            edge_set.discard(new_e1)
            edge_set.discard(new_e2)
            edge_set.add(e1)
            edge_set.add(e2)
            
            edges.remove(new_e1)
            edges.remove(new_e2)
            edges.append(e1)
            edges.append(e2)
        
        temp *= cooling
    
    return best_edges, best_count

def _try_cage_320(time_left):
    """Try to find a 320-vertex (4,10)-cage."""
    n = 320
    rng = random.Random(12345)
    
    # Try multiple random initializations
    for attempt in range(20):
        if time_left < 5:
            break
        
        edges = _random_4_regular(n, seed=1000 + attempt)
        if edges is None:
            continue
        
        # Check if already valid
        if _validate_graph(n, edges, 4, 10):
            return n, edges
        
        # Try to improve
        remaining = time_left - 5
        if remaining < 3:
            break
        
        anneal_time = min(remaining * 0.7, 30.0)
        improved, count = _improve_girth(n, edges, 10, anneal_time, rng)
        
        if count == 0:
            if _validate_graph(n, improved, 4, 10):
                return n, improved
    
    return None, None

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    if k == 4 and g == 10:
        # Try to find the 320-vertex cage
        time_left = deadline - time.time()
        n_cage, edges_cage = _try_cage_320(time_left)
        
        if n_cage is not None and n_cage < best_n:
            best_n = n_cage
            best_edges = edges_cage
    
    return best_n, best_edges
