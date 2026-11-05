"""Improved program for cage problem n(5,9)."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _bfs_check_girth(adj, start, max_dist):
    """BFS from start, return distance to all reachable vertices.
    Used to check girth: if we find a vertex at distance d from start
    that is also adjacent to a vertex at distance d-1 (other than parent),
    that indicates a cycle."""
    dist = {start: 0}
    parent = {start: -1}
    queue = [start]
    for i in range(len(queue)):
        u = queue[i]
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
                if dist[v] > max_dist:
                    return dist
    return dist

def _check_cycle_created(adj, u, v, g):
    """Check if adding edge (u,v) to graph adj creates a cycle of length < g.
    Returns True if a short cycle is created, False otherwise."""
    # A cycle of length L < g through (u,v) means there's a path from u to v
    # of length L-1 in the existing graph (not using edge (u,v)).
    # So we need to check if dist(u,v) < g-1 in the current graph.
    
    # BFS from u, stop when we reach depth g-2 (since we need path length <= g-2 to form cycle length <= g-1)
    max_depth = g - 2  # path of length max_depth gives cycle of length max_depth+1
    if max_depth < 1:
        return False
    
    dist = {u: 0}
    queue = [(u, 0)]
    for i in range(len(queue)):
        node, d = queue[i]
        if d >= max_depth:
            continue
        for neighbor in adj[node]:
            if neighbor == v:
                # Found path from u to v of length d+1, cycle length d+2
                if d + 2 < g:
                    return True
            if neighbor not in dist:
                dist[neighbor] = d + 1
                queue.append((neighbor, d + 1))
    return False

def _build_random_greedy(k, g, deadline):
    """Build a k-regular graph of girth at least g using random greedy."""
    # Moore bound for k=5, g=9: 1 + k*(1 + k + k^2 + ... + k^(g/2 - 1)) for even g
    # For odd g, Moore bound is different. Let's use a reasonable lower bound.
    # For k=5, g=9: lower bound ~ 1 + 5*(1 + 5 + 25) = 1 + 5*31 = 156
    # But best known is 1116. Let's try to build something in that range.
    
    # Start with a small number and grow
    min_n = 100  # lower bound attempt
    max_n = 1200  # upper bound
    
    best_n = None
    best_edges = None
    
    start_time = time.time()
    
    for n in range(min_n, min(max_n + 1, 1500)):
        if time.time() > deadline - 10:
            break
        
        # Try to build a k-regular graph on n vertices
        adj = [[] for _ in range(n)]
        degree = [0] * n
        
        # Shuffle vertices for randomness
        vertices = list(range(n))
        random.shuffle(vertices)
        
        success = True
        for v in vertices:
            # Try to add k edges from v
            attempts = 0
            while degree[v] < k and attempts < 500:
                u = random.randint(0, n - 1)
                if u == v:
                    attempts += 1
                    continue
                if u in adj[v]:
                    attempts += 1
                    continue
                if degree[u] >= k:
                    attempts += 1
                    continue
                # Check if adding (v,u) creates a short cycle
                if _check_cycle_created(adj, v, u, g):
                    attempts += 1
                    continue
                # Add edge
                adj[v].append(u)
                adj[u].append(v)
                degree[v] += 1
                degree[u] += 1
                attempts = 0
            
            if degree[v] < k:
                success = False
                break
        
        if success:
            # Verify the graph is valid: k-regular and girth >= g
            # Check k-regular
            if all(len(adj[i]) == k for i in range(n)):
                # Check girth: for each vertex, BFS to verify no cycle < g
                girth_ok = True
                for start in range(n):
                    dist = {start: 0}
                    parent = {start: -1}
                    queue = [start]
                    for i in range(len(queue)):
                        u = queue[i]
                        for v in adj[u]:
                            if v not in dist:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                # Check if this creates a cycle
                                if v != parent[u] and u != parent.get(v, -1):
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < g:
                                        girth_ok = False
                                        break
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
        
        if best_n is not None and n >= best_n:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    return None, None

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    return len(order), edges

def construct(k, g):
    deadline = time.time() + 115  # 5s safety margin
    random.seed(42)
    
    # Try random greedy first
    result = _build_random_greedy(k, g, deadline)
    if result[0] is not None:
        return result
    
    # Fallback to LUW
    return _luw_construct(k, g)
# EVOLVE-BLOCK-END
