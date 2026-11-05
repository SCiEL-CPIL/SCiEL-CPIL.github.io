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

def _random_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices.
    """
    rng = random.Random(seed)
    
    for attempt in range(100):
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

def _compute_cost(adj_np, n, g):
    """
    Compute cost as sum of traces of A^k for k=3 to g-1.
    """
    cost = 0.0
    A = adj_np
    # A^1 is adj
    # We need traces of A^3 to A^(g-1)
    # A^2 = A @ A
    A2 = A @ A
    # A^3 = A2 @ A
    A3 = A2 @ A
    cost += np.trace(A3)
    
    # A^4 = A2 @ A2
    A4 = A2 @ A2
    cost += np.trace(A4)
    
    # A^5 = A4 @ A
    A5 = A4 @ A
    cost += np.trace(A5)
    
    # A^6 = A3 @ A3
    A6 = A3 @ A3
    cost += np.trace(A6)
    
    # A^7 = A6 @ A
    A7 = A6 @ A
    cost += np.trace(A7)
    
    # A^8 = A4 @ A4
    A8 = A4 @ A4
    cost += np.trace(A8)
    
    # A^9 = A8 @ A
    A9 = A8 @ A
    cost += np.trace(A9)
    
    return cost

def _improve_girth_fast(n, edges, g, time_limit, rng):
    """
    Simulated annealing with trace-based cost.
    """
    adj_np = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        adj_np[u, v] = 1
        adj_np[v, u] = 1
        
    current_cost = _compute_cost(adj_np, n, g)
    if current_cost == 0:
        return edges, 0
        
    best_edges = list(edges)
    best_cost = current_cost
    
    t_start = time.time()
    temp = 50.0
    cooling = 0.999
    
    edge_list = list(edges)
    
    while time.time() < t_start + time_limit:
        e1 = rng.choice(edge_list)
        e2 = rng.choice(edge_list)
        if e1 == e2:
            continue
            
        u1, v1 = e1
        u2, v2 = e2
        
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        if adj_np[u1, u2] == 1 or adj_np[v1, v2] == 1:
            continue
            
        adj_np[u1, v1] = 0
        adj_np[v1, u1] = 0
        adj_np[u2, v2] = 0
        adj_np[v2, u2] = 0
        
        adj_np[u1, u2] = 1
        adj_np[u2, u1] = 1
        adj_np[v1, v2] = 1
        adj_np[v2, v1] = 1
        
        edge_list.remove(e1)
        edge_list.remove(e2)
        new_e1 = (min(u1, u2), max(u1, u2))
        new_e2 = (min(v1, v2), max(v1, v2))
        edge_list.append(new_e1)
        edge_list.append(new_e2)
        
        new_cost = _compute_cost(adj_np, n, g)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_edges = list(edge_list)
            
        if new_cost <= current_cost or rng.random() < pow(2.718, -(new_cost - current_cost) / temp):
            current_cost = new_cost
        else:
            adj_np[u1, u2] = 0
            adj_np[u2, u1] = 0
            adj_np[v1, v2] = 0
            adj_np[v2, v1] = 0
            
            adj_np[u1, v1] = 1
            adj_np[v1, u1] = 1
            adj_np[u2, v2] = 1
            adj_np[v2, u2] = 1
            
            edge_list.remove(new_e1)
            edge_list.remove(new_e2)
            edge_list.append(e1)
            edge_list.append(e2)
            
        temp *= cooling
        
    return best_edges, best_cost

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    if k == 4 and g == 10:
        target_n = 320
        rng = random.Random(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            edges = _random_4_regular(target_n, seed=100 + attempt)
            if edges is None:
                continue
                
            if _validate_graph(target_n, edges, k, g):
                best_n = target_n
                best_edges = edges
                break
                
            time_left = deadline - time.time()
            if time_left < 5:
                break
                
            anneal_time = time_left * 0.8
            improved_edges, cost = _improve_girth_fast(target_n, edges, g, anneal_time, rng)
            
            if cost == 0:
                if _validate_graph(target_n, improved_edges, k, g):
                    best_n = target_n
                    best_edges = improved_edges
                    break

    return best_n, best_edges
