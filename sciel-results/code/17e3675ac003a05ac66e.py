import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Use Lazebnik-Ustimenko-Woldar (LUW) graph CD(d, q) with q=k.
    # Girth is 2d+2 if q >= d+1.
    # We want girth >= g.
    # So 2d+2 >= g => d >= (g-2)/2.
    # We iterate d starting from min_d and check validity.
    
    min_d = max(3, (g - 2 + 1) // 2)
    
    best_n = float('inf')
    best_edges = None

    # Try a range of d values
    for d in range(min_d, min_d + 10):
        if time.time() > deadline - 30.0:
            break
            
        # Check if k is suitable for LUW (prime or 4)
        if k != 4 and not _is_prime(k):
            continue
            
        n, edges = _luw_graph(k, d, deadline)
        if n is None:
            continue
            
        # Verify girth
        if _verify_girth(n, edges, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            break # Smallest d gives smallest n, so we can stop if valid

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: Original seed logic (which might work for non-prime k or if LUW fails)
    return _original_construct(k, g, deadline)

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _luw_graph(k, d, deadline):
    q = k
    
    # Setup field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW graph
    # rule[c] = (a, b) such that y[c] = y[a] * y[b] + y[c-1] ?
    # The seed code uses a specific rule. I will replicate it.
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            return None, None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _verify_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles shorter than g
    # Since LUW is bipartite, cycles are even.
    # We check for cycles of length 4, 6, ..., up to g-1 (if g is even) or g-2 (if g is odd? No, girth < g).
    # Max cycle length to check: L_max = g - 1 if g is odd, g - 2 if g is even?
    # Actually, just check all even lengths < g.
    max_len = g - 1
    if max_len % 2 == 1:
        max_len -= 1
        
    if max_len < 4:
        return True # No cycles shorter than 4 possible in simple graph with g>=5? 
        # Actually if g=3, we check length 2 (impossible) and 3 (odd, not in bipartite).
        # So for bipartite, min cycle is 4.
        
    # BFS from each vertex up to depth max_len // 2
    # If we return to start, we found a cycle.
    
    for start in range(n):
        # visited[depth][node] = parent
        # We only need to track depth and parent to avoid backtracking
        # But for small depths, we can just use a set of (node, depth) or dict node->depth
        # Since graph is bipartite, we can use two sets or just BFS with depth tracking.
        
        # We want to detect if start is reached at depth d > 0.
        # BFS:
        # Queue: (node, depth, parent)
        # If node == start and depth > 0:
        #    Cycle length = 2 * depth.
        #    If 2 * depth < g: return False.
        
        visited = {start: 0} # node -> depth
        queue = deque([(start, 0, -1)])
        
        while queue:
            node, depth, parent = queue.popleft()
            if depth >= max_len // 2:
                continue
                
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == start:
                    cycle_len = 2 * (depth + 1)
                    if cycle_len < g:
                        return False
                    # If cycle_len >= g, it's okay.
                elif neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    # If neighbor is visited, it might be a cycle.
                    # But in bipartite graph, we only care about returning to start?
                    # No, any cycle is a problem.
                    # However, if we found a cycle not involving start, it would have been found
                    # when starting BFS from a vertex on that cycle.
                    # So checking return to start is sufficient for existence of ANY short cycle.
                    pass
                    
    return True

def _original_construct(k, g, deadline):
    # Fallback to seed code logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

# EVOLVE-BLOCK-END
