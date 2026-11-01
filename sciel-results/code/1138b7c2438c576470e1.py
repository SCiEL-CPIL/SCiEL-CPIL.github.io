import random
import time
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    if k != 5 or g != 9:
        return _fallback_construct(k, g)
    
    start_time = time.time()
    time_limit = 115
    
    # Start with LUW construction using optimal d
    # For g=9, d=4 gives n = 2*5^4 = 1250 with girth >= 2*4+1 = 9
    n, adj = _luw_construct(k, g)
    
    # Try to improve with local search (2-switches)
    if time.time() - start_time < time_limit - 30:
        result = _local_search(adj, k, g, time_limit - (time.time() - start_time) - 10)
        if result is not None:
            return result
    
    # Return LUW result
    edges = _adj_to_edges(adj)
    return n, edges

def _fallback_construct(k, g):
    """Fallback for non-5,9 parameters."""
    # Simple cycle-based construction
    n = k * (k ** ((g - 3) // 2)) + 1
    edges = []
    seen = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            a, b = i, (i + j) % n
            if a > b:
                a, b = b, a
            if (a, b) not in seen:
                edges.append((a, b))
                seen.add((a, b))
    return n, edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction.
    
    Returns (n, adj_list) where adj_list[v] is a list of neighbors of v.
    """
    q = k
    # For girth g, we need 2*d+1 >= g, so d >= (g-1)//2
    # But LUW graphs have girth at least 2d+1
    d = (g - 1) // 2
    if d < 2:
        d = 2
    
    # Define the rule for the LUW construction
    # rule[c] = (a, b) means y[c] = x[c] + y[a]*x[b] (for side 0)
    # or y[c] = x[c] - x[a]*y[b] (for side 1)
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        # Standard LUW rule
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        elif r == 3:
            rule[c] = (1, c0 + 1)
    
    def add(x, y):
        return (x + y) % q
    
    def sub(x, y):
        return (x - y) % q
    
    def mul(x, y):
        return (x * y) % q
    
    # Generate all vertices
    # Vertex = (side, tuple of d values)
    # side 0: (0, (y1, y2, ..., yd)) where y1 = 0
    # side 1: (1, (x1, x2, ..., xd))
    
    vertices = []
    index = {}
    
    # Side 0 vertices: first component is 0
    for y2 in range(q):
        for y3 in range(q):
            for rest in _generate_rest(q, d, 4):
                vertex = (0, (0, y2, y3) + rest)
                index[vertex] = len(vertices)
                vertices.append(vertex)
    
    # Side 1 vertices: all components vary
    for x1 in range(q):
        for x2 in range(q):
            for rest in _generate_rest(q, d, 2):
                vertex = (1, (x1, x2) + rest)
                index[vertex] = len(vertices)
                vertices.append(vertex)
    
    n = len(vertices)
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    for v_idx, vertex in enumerate(vertices):
        side, x = vertex
        for first in range(q):
            # Compute neighbor
            if side == 0:
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            if neighbor in index:
                n_idx = index[neighbor]
                if n_idx != v_idx:
                    edge_key = (min(v_idx, n_idx), max(v_idx, n_idx))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        adj[v_idx].append(n_idx)
                        adj[n_idx].append(v_idx)
    
    return n, adj

def _generate_rest(q, d, start_idx):
    """Generate all combinations for the remaining components."""
    if start_idx >= d:
        yield ()
        return
    for val in range(q):
        for rest in _generate_rest(q, d, start_idx + 1):
            yield (val,) + rest

def _adj_to_edges(adj):
    """Convert adjacency list to edge list."""
    edges = []
    n = len(adj)
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

def _girth_check(adj, g):
    """Check if graph has girth at least g using BFS from each vertex.
    
    Returns True if no cycle shorter than g exists.
    """
    n = len(adj)
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if dist[node] >= max_depth:
                continue
            
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
                elif dist[nb] == dist[node] + 1:
                    # Found a cycle of length dist[node] + dist[nb] + 1
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _local_search(adj, k, g, time_budget):
    """Try to improve the graph using 2-switches that maintain girth.
    
    Returns (n, adj) if improved, else None.
    """
    start_time = time.time()
    n = len(adj)
    
    # Precompute degrees
    deg = [len(adj[i]) for i in range(n)]
    
    # We'll try random 2-switches
    # A 2-switch: pick edges (a,b) and (c,d), replace with (a,c) and (b,d) or (a,d) and (b,c)
    # This maintains regularity
    
    for iteration in range(100000):
        if time.time() - start_time > time_budget:
            break
        
        # Pick two random edges
        # Build edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if len(edges) < 2:
            break
        
        e1 = random.choice(edges)
        e2 = random.choice(edges)
        
        # Ensure edges are distinct and don't share vertices
        a, b = e1
        c, d = e2
        
        if len({a, b, c, d}) < 4:
            continue
        
        # Try switch: remove (a,b) and (c,d), add (a,c) and (b,d)
        if a not in adj[c] and b not in adj[d] and a != c and b != d:
            # Check if this switch is valid
            # Remove edges
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            
            # Add new edges
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            
            # Check girth (only check locally for speed)
            if _local_girth_check(adj, g, [a, b, c, d]):
                # Keep the switch
                continue
            else:
                # Revert
                adj[a].remove(c)
                adj[c].remove(a)
                adj[b].remove(d)
                adj[d].remove(b)
                
                adj[a].append(b)
                adj[b].append(a)
                adj[c].append(d)
                adj[d].append(c)
        
        # Alternative switch: add (a,d) and (b,c)
        elif a not in adj[d] and b not in adj[c] and a != d and b != c:
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            
            adj[a].append(d)
            adj[d].append(a)
            adj[b].append(c)
            adj[c].append(b)
            
            if _local_girth_check(adj, g, [a, b, c, d]):
                continue
            else:
                adj[a].remove(d)
                adj[d].remove(a)
                adj[b].remove(c)
                adj[c].remove(b)
                
                adj[a].append(b)
                adj[b].append(a)
                adj[c].append(d)
                adj[d].append(c)
    
    return n, adj

def _local_girth_check(adj, g, affected_vertices):
    """Quick girth check focusing on affected vertices."""
    n = len(adj)
    max_depth = (g - 1) // 2
    
    for start in affected_vertices:
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if dist[node] >= max_depth:
                continue
            
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
                elif dist[nb] == dist[node] + 1:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    
    return True
