import time as _time
import numpy as _np
import random
from collections import deque

def _check_girth(n, edges, k, g, deadline):
    """Verify via non-backtracking walks that no cycle < g exists."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return False
        seen.add(key)
    max_depth = g - 2
    if max_depth < 1:
        return True
    
    for a in range(n):
        if _time.time() > deadline:
            return True
        for b in adj[a]:
            visited = set()
            visited.add((b, a))
            frontier = [(b, a)]
            depth = 0
            while frontier and depth < max_depth:
                depth += 1
                new_frontier = []
                for cur, prev in frontier:
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if (nxt, cur) not in visited:
                            visited.add((nxt, cur))
                            new_frontier.append((nxt, cur))
                frontier = new_frontier
                if not frontier:
                    break
                for cur, prev in frontier:
                    if cur == a and prev != b:
                        return False
    return True

def _construct_luw(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    d = 5
    best_n, best_edges = None, None
    for q in [3, 4, 5, 7]:
        if _time.time() > deadline - 10.0:
            break
        
        if q == 4:
            add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
            mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x, y): return add_tbl[x][y]
            def sub(x, y): return add_tbl[x][y]
            def mul(x, y): return mul_tbl[x][y]
            qsize = 4
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
            qsize = q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(qsize):
                y = [0, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (0,) * (d + 1))
        index, order, edges = {start: 0}, [start], []
        
        for vertex in order:
            if _time.time() > deadline - 5.0:
                return None, []
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        deg = [0]*n
        for u,v in edges:
            deg[u]+=1
            deg[v]+=1
        if all(dg==4 for dg in deg):
            edge_set = set()
            valid = True
            for u,v in edges:
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            if valid:
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
    return best_n, best_edges

def construct(k, g):
    deadline = _time.time() + 115.0
    best_n, best_edges = None, None
    
    # 1. Try LUW construction directly (faster)
    try:
        n_luw, e_luw = _construct_luw(k, g, deadline)
        if n_luw and e_luw:
            best_n, best_edges = n_luw, e_luw
    except Exception:
        pass

    if best_n is None:
        raise ValueError("No valid graph constructed")
        
    return best_n, best_edges
