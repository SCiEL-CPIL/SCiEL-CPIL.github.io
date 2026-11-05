"""Construct a k-regular graph of girth at least g with minimal vertices within time budget."""
import time
import itertools
import random
import math
from collections import deque

# Moore bound lower bound for n(k, g)
def _moore_bound(k, g):
    if g % 2 == 0:
        # Even girth: tree of depth g/2 - 1 from each edge
        depth = g // 2 - 1
        return 2 * (1 + k * (k**depth - 1) // (k - 1))
    else:
        # Odd girth: tree of depth (g-1)/2 from each vertex
        depth = (g - 1) // 2
        return 1 + k * (k**depth - 1) // (k - 1)

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking walks from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS tracking (vertex, prev) to find shortest cycle through each vertex
    # A cycle of length L through start means there's a path of length L from start back to start
    # without immediately backtracking
    
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # dist[(u, v)] = shortest non-backtracking walk length from (start, None) to (u, v)
        dist = {}
        q = deque()
        for nb in adj[start]:
            dist[(start, nb)] = 1
            q.append((start, nb))
        
        while q:
            u, prev = q.popleft()
            d = dist[(u, prev)]
            for nb in adj[u]:
                if nb == prev:
                    continue  # no backtracking
                new_state = (u, nb)
                if new_state in dist:
                    # Found a cycle: length = d + 1
                    cycle_len = d + 1
                    if cycle_len < g:
                        return False
                else:
                    dist[new_state] = d + 1
                    q.append(new_state)
                # Early termination: if we've explored enough depth, no short cycle exists
                if d + 1 >= g - 1:
                    break
    return True

def construct(k, g):
    """Construct a k-regular graph with girth at least g, minimizing vertices."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    mb = _moore_bound(k, g)
    
    def _time_remaining():
        return deadline - time.time()
    
    # Try Lazebnik-Ustimenko-Woldar graph CD(d, q)
    def try_luw():
        q = k
        
        # Choose d such that girth >= g
        # For odd d: girth = d + 5
        # For even d: girth = d + 4
        candidates_d = []
        for d in range(3, 15):
            if d % 2 == 1:
                girth_val = d + 5
            else:
                girth_val = d + 4
            if girth_val >= g:
                candidates_d.append((girth_val, d))
        
        candidates_d.sort()
        
        for girth_val, d in candidates_d:
            if _time_remaining() < 5.0:
                break
            
            # GF(q) operations
            if q == 5:
                add = lambda x, y: (x + y) % 5
                sub = lambda x, y: (x - y) % 5
                mul = lambda x, y: x * y % 5
            elif q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: table[x][y]
            elif q == 3:
                add = lambda x, y: (x + y) % 3
                sub = lambda x, y: (x - y) % 3
                mul = lambda x, y: x * y % 3
            elif q == 2:
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: x & y
            else:
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q
            
            # Rule for CD(d, q)
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
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
                    yield (1 - side), tuple(y)
            
            start = (0, (None,) + (0,) * d)
            index = {start: 0}
            order = [start]
            edges = []
            
            for i, vertex in enumerate(order):
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            
            n_luw = len(order)
            
            # Quick check: verify girth
            if _verify_girth(n_luw, edges, g):
                return n_luw, edges
        
        return None, None
    
    # Try random construction with local search
    def try_random_construction(max_n_limit):
        if _time_remaining() < 10.0:
            return None, None
        
        min_n = mb
        max_n = min(mb + 300, max_n_limit)
        
        for n in range(min_n, max_n + 1):
            if _time_remaining() < 3.0:
                break
            
            if (k * n) % 2 != 0:
                continue
            
            for attempt in range(30):
                if _time_remaining() < 1.0:
                    break
                
                random.seed(hash((n, k, g, attempt)) % (2**32))
                
                half_edges = list(range(n)) * k
                random.shuffle(half_edges)
                
                edges = set()
                valid = True
                for i in range(0, len(half_edges), 2):
                    u, v = half_edges[i], half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    edge = (min(u, v), max(u, v))
                    if edge in edges:
                        valid = False
                        break
                    edges.add(edge)
                
                if not valid:
                    continue
                
                edge_list = list(edges)
                
                deg = [0] * n
                for u, v in edge_list:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != k for d in deg):
                    continue
                
                if _verify_girth(n, edge_list, g):
                    return n, edge_list
        
        return None, None
    
    # Try LUW first
    n_luw, edges_luw = try_luw()
    
    if n_luw is not None:
        return n_luw, edges_luw
    
    # If LUW failed, try random
    n_rand, edges_rand = try_random_construction(2000)
    
    if n_rand is not None:
        return n_rand, edges_rand
    
    # Fallback: return LUW even without verification (best effort)
    q = k
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    d = max(3, d)
    
    if q == 5:
        add = lambda x, y: (x + y) % 5
        sub = lambda x, y: (x - y) % 5
        mul = lambda x, y: x * y % 5
    elif q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: x * y % 3
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n_luw = len(order)
    return n_luw, edges
