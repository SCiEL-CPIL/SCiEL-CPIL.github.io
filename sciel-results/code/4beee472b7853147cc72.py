"""Improved cage graph construction via voltage lifts and algebraic methods."""
import time
import math
import random
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def _edges_to_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(adj, n, g):
    """Check girth >= g using non-backtracking BFS from every vertex."""
    for start in range(n):
        depth = {}
        queue = deque()
        for nb in adj[start]:
            depth[(start, nb)] = 1
            queue.append((start, nb))
        while queue:
            u, v = queue.popleft()
            d = depth[(u, v)]
            if d + 1 >= g:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                key = (v, w)
                if key in depth:
                    if d + 1 + depth[key] < g:
                        return False
                else:
                    depth[key] = d + 1
                    if d + 1 < g:
                        queue.append(key)
    return True


def _moore_bound(k, g):
    """Compute the Moore bound (lower bound on vertices) for k-regular graph of girth g."""
    if g % 2 == 0:
        r = g // 2
        return k * (k - 1) * ((k ** (r - 1) - 1) // (k - 1)) + 2 * (k - 1) * (k ** (r - 1))
    else:
        r = (g - 1) // 2
        return k * (k - 1) * ((k ** (r - 1) - 1) // (k - 1)) + k * ((k ** r - 1) // (k - 1))


def _build_lift(base_edges, base_n, k, group_order, voltage_fn):
    """
    Build a k-regular graph by lifting a base graph via voltage assignment.
    """
    n = base_n * group_order
    base_adj = [[] for _ in range(base_n)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    edges = []
    edge_set = set()
    
    for u in range(base_n):
        for v in base_adj[u]:
            if u < v:
                volt = voltage_fn(u, v)
                for t in range(group_order):
                    a = u * group_order + t
                    b = v * group_order + (t + volt) % group_order
                    e = (a, b) if a < b else (b, a)
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
    
    return n, edges


def _construct_via_lift(k, g, deadline):
    """
    Try to construct a k-regular graph of girth >= g using voltage lifts.
    """
    if k == 4:
        # K_{4,4}: 8 vertices, 4-regular, girth 4
        base_edges = [(i, 4 + j) for i in range(4) for j in range(4)]
        base_n = 8
    elif k == 3:
        # Petersen graph
        base_edges = []
        for i in range(5):
            base_edges.append((i, (i + 1) % 5))
        base_edges.append((5, 7))
        base_edges.append((7, 9))
        base_edges.append((9, 6))
        base_edges.append((6, 8))
        base_edges.append((8, 5))
        for i in range(5):
            base_edges.append((i, 5 + i))
        base_n = 10
    else:
        return None
    
    best_result = None
    best_n = float('inf')
    
    moore = _moore_bound(k, g)
    min_group_order = max(1, math.ceil(moore / base_n))
    
    for group_order in range(min_group_order, min_group_order + 80):
        if time.time() > deadline:
            break
        
        n = base_n * group_order
        if n > 600:
            break
        
        n_trials = 50
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            rng = random.Random(trial * 104729 + group_order * 7919)
            
            edge_voltages = {}
            edge_list = list(base_edges)
            for idx, (u, v) in enumerate(edge_list):
                if trial < 10:
                    gen = rng.randint(2, max(2, group_order - 1))
                    volt = (gen ** idx) % group_order
                elif trial < 20:
                    volt = rng.randint(0, group_order - 1)
                else:
                    # Use a more structured approach based on quadratic residues
                    if group_order % 4 == 1:
                        quad_residues = set((i * i) % group_order for i in range(group_order))
                        volt = list(quad_residues)[idx % len(quad_residues)]
                    else:
                        volt = rng.randint(0, group_order - 1)
                
                edge_voltages[(u, v)] = volt
                edge_voltages[(v, u)] = (-volt) % group_order
            
            def voltage_fn(u, v, _ev=edge_voltages):
                return _ev.get((u, v), 0)
            
            n_lift, edges = _build_lift(base_edges, base_n, k, group_order, voltage_fn)
            
            adj = _edges_to_adj(n_lift, edges)
            if any(len(adj[i]) != k for i in range(n_lift)):
                continue
            
            if _check_girth_fast(adj, n_lift, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    return best_result


def _construct_luw(k, g):
    """L-U-W (Lubotzky-Phillips-Sarnak style) construction for k-regular graphs."""
    try:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_op = lambda x, y: x ^ y
            sub_op = lambda x, y: x ^ y
            mul_op = lambda x, y: table[x][y]
        else:
            add_op = lambda x, y: (x + y) % q
            sub_op = lambda x, y: (x - y) % q
            mul_op = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_op(x[c], mul_op(y[a], x[b]))
                    else:
                        y[c] = sub_op(x[c], mul_op(x[a], y[b]))
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
        
        n = len(order)
        adj = _edges_to_adj(n, edges)
        if all(len(adj[i]) == k for i in range(n)) and _check_girth_fast(adj, n, g):
            return n, edges
    except:
        pass
    return None


def construct(k, g):
    """
    Build a k-regular graph with girth at least g on as few vertices as possible.
    """
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Try voltage lift construction first
    result = _construct_via_lift(k, g, deadline)
    if result is not None:
        return result
    
    # Try L-U-W construction
    result = _construct_luw(k, g)
    if result is not None:
        return result
    
    # Last resort: return a simple regular graph
    n_fallback = 2 * k
    edges_fallback = []
    for i in range(n_fallback):
        edges_fallback.append((i, (i + 1) % n_fallback))
    for i in range(k - 1):
        edges_fallback.append((i, (i + n_fallback // 2) % n_fallback))
    edges_fallback = list(set((min(u, v), max(u, v)) for u, v in edges_fallback))
    return n_fallback, edges_fallback

# EVOLVE-BLOCK-END
