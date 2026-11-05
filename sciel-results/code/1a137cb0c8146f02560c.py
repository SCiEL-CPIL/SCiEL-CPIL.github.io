"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import numpy as np

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    # Search for 4-regular girth-g Cayley graphs on Z_n x Z_2
    # Generators: (1,0), (-1,0), (a,1), (-a,1)
    # Order: 2n
    # Moore bound for (4,10) is 243, so 2n >= 243 => n >= 122
    
    best_n = None
    best_edges = None
    
    max_n_search = 500
    
    for n in range(122, max_n_search + 1):
        found = False
        for a in range(1, n):
            # Check distinctness
            if (2 * a) % n == 0:
                continue
            
            if _check_girth_ab2(n, a, g):
                edges = set()
                for i in range(n):
                    for z in [0, 1]:
                        v = (i, z)
                        u1 = ((i+1)%n, z)
                        u2 = ((i-1)%n, z)
                        u3 = ((i+a)%n, 1-z)
                        u4 = ((i-a)%n, 1-z)
                        
                        for u in [u1, u2, u3, u4]:
                            if v < u:
                                edges.add((v, u))
                            else:
                                edges.add((u, v))
                
                edges_list = sorted(edges)
                total_vertices = 2 * n
                if best_n is None or total_vertices < best_n:
                    best_n = total_vertices
                    best_edges = edges_list
                found = True
                break
        if found:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    return _construct_fallback(4, g)


def _check_girth_ab2(n, a, g):
    S = [(1, 0), (-1, 0), (a, 1), (-a, 1)]
    inv = {(1, 0): (-1, 0), (-1, 0): (1, 0), (a, 1): (-a, 1), (-a, 1): (a, 1)}
    
    for L in range(3, g):
        def check_walk(s1, idx, pos, last):
            if idx == L:
                if pos == (0, 0):
                    if last != inv[s1]:
                        return True
                    else:
                        return False
                return False
            
            for s in S:
                if s == inv[last]:
                    continue
                new_pos = ((pos[0] + s[0]) % n, (pos[1] + s[1]) % 2)
                if check_walk(s1, idx + 1, new_pos, s):
                    return True
            return False
        
        for s1 in S:
            if check_walk(s1, 2, s1, s1):
                return False
    return True

def _verify_girth(adj, n, g):
    for start in range(n):
        visited = {start: (-1, 0)}
        queue = [(start, -1, 0)]
        while queue:
            node, parent, dist = queue.pop(0)
            if dist >= g:
                continue
            for nei in adj[node]:
                if nei == parent:
                    continue
                if nei in visited:
                    cycle_len = dist + visited[nei][1] + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[nei] = (node, dist + 1)
                    queue.append((nei, node, dist + 1))
    return True


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    while True:
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
        
        n = len(order)
        
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        if _verify_girth(adj, n, g):
            return n, edges
        else:
            d += 1
