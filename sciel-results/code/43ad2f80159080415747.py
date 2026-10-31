import time
import random
import numpy as np
from collections import deque

def _check_girth(adj, n, g):
    """Check if graph has girth >= g using BFS from each vertex with non-backtracking."""
    for start in range(n):
        # BFS tracking (node, parent) to avoid backtracking
        # dist[v] = shortest non-backtracking distance from start
        dist = [-1] * n
        dist[start] = 0
        queue = deque([(start, -1)])
        while queue:
            node, parent = queue.popleft()
            d = dist[node]
            if d + 1 >= g:
                break
            for nb in adj[node]:
                if nb == parent:
                    continue
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    queue.append((nb, node))
                else:
                    # Found a cycle: dist[node] + 1 + dist[nb] ... 
                    # Actually, the cycle length is dist[node] + 1 + dist[nb] only if 
                    # the path from start to nb doesn't go through node. 
                    # This is tricky. Let me use a different approach.
                    # The cycle length is at most d + 1 + dist[nb]
                    # But we need to be careful.
                    cycle_len = d + 1 + dist[nb]
                    if cycle_len < g:
                        return False
        # Also check: if we found a neighbor at distance that creates a short cycle
        # The above check handles it when we encounter an already-visited node
    return True

def _check_girth_fast(adj, n, g):
    """Faster girth check: for each edge, check if there's a path of length < g-2 between its endpoints avoiding the edge."""
    # Actually, let's use a simpler but correct approach:
    # For each vertex, do BFS and track distances. If we find a back-edge that creates a cycle < g, return False.
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d + 1 >= g:
                break
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    parent[nb] = node
                    queue.append(nb)
                else:
                    # This edge (node, nb) where nb is already visited and nb != parent[node]
                    # creates a cycle of length dist[node] + 1 + dist[nb]
                    # But this is only valid if the paths are vertex-disjoint except at start
                    # In a BFS tree, the path from start to node and start to nb might share vertices.
                    # The cycle length is at most dist[node] + 1 + dist[nb]
                    cycle_len = dist[node] + 1 + dist[nb]
                    if cycle_len < g:
                        return False
    return True

def _generate_random_3reg(n, seed=None):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if seed is not None:
        random.seed(seed)
    if n % 2 != 0:
        return None
    # Configuration model: each vertex has 3 half-edges
    half_edges = []
    for v in range(n):
        for _ in range(3):
            half_edges.append(v)
    random.shuffle(half_edges)
    # Pair up half-edges
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            continue  # Skip loops, but this breaks 3-regularity... 
        # This approach is flawed. Let me use a different method.
    return None

def _generate_random_3reg_v2(n, seed=None):
    """Generate random 3-regular graph using edge swaps from a base graph."""
    if seed is not None:
        random.seed(seed)
    if n < 4 or n % 2 != 0:
        return None
    # Start with a cycle and add chords to make it 3-regular
    # For n vertices, a cycle gives degree 2. We need n/2 more edges.
    # Use a random 3-regular graph generation.
    
    # Use the configuration model properly
    half_edges = list(range(n)) * 3
    random.shuffle(half_edges)
    
    adj = [set() for _ in range(n)]
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
        # If there's a loop or duplicate, we'd need to retry, but let's check later
    
    # Check if all vertices have degree 3
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    
    # Convert to list
    adj_list = [list(s) for s in adj]
    return adj_list

def _swap_edges(adj, n, seed=None):
    """Try one random edge swap and return new adjacency if valid."""
    if seed is not None:
        random.seed(seed)
    # Pick two random edges and try to swap
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if len(edges) < 2:
        return None
    
    (u1, v1), (u2, v2) = random.sample(edges, 2)
    
    # Try swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
    # or (u1,v2) and (v1,u2)
    for new_edges in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
        (a, b), (c, d) = new_edges
        if a == b or c == d:
            continue
        if (min(a,b), max(a,b)) == (u1, v1) or (min(a,b), max(a,b)) == (u2, v2):
            continue
        if (min(c,d), max(c,d)) == (u1, v1) or (min(c,d), max(c,d)) == (u2, v2):
            continue
        if (min(a,b), max(a,b)) == (min(c,d), max(c,d)):
            continue
        
        # Check no duplicates
        if b in adj[a] and d in adj[c]:
            continue
        if d in adj[a] and b in adj[c]:
            continue
        
        # Make the swap
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        adj[a].append(b)  # Wait, we're adding (a,b) which we just removed... 
        # Let me redo this.
        # Remove (u1,v1) and (u2,v2)
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        # Add (a,b) and (c,d)
        adj[a].append(b)
        adj[b].append(a)
        adj[c].append(d)
        adj[d].append(c)
        return adj
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore bound for 3-regular, girth g
    if g % 2 == 0:
        # Even girth: Moore bound is 2 * sum_{i=0}^{g/2-1} (k-1)^i
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        # Odd girth: Moore bound is 1 + k * sum_{i=0}^{(g-3)//2} (k-1)^i
        mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = max(mb, 4)
    if mb % 2 != 0:
        mb += 1
    
    best_n = None
    best_edges = None
    
    # Try sizes from Moore bound upward
    n = mb
    max_n = 2000  # Limit to keep runtime manageable
    attempt = 0
    
    while n <= max_n and time.time() < deadline:
        attempt += 1
        # Try multiple random graphs for this size
        max_attempts = 50 if n < 600 else 20
        for trial in range(max_attempts):
            if time.time() > deadline - 5:
                break
            adj = _generate_random_3reg_v2(n, seed=attempt * 1000 + trial)
            if adj is None:
                continue
            if _check_girth_fast(adj, n, g):
                # Found a valid graph!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                return n, edges
        n += 2  # Must be even for 3-regular
    
    # If we didn't find anything, fall back to LUW construction
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
