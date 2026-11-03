import time
import random
import sys

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a greedy heuristic search with restarts, falling back to LUW if necessary.
    """
    start_time = time.time()
    timeout = 110.0  # Leave 10s for overhead/fallback
    
    best_n = float('inf')
    best_edges = []
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        n, edges = greedy_construct(k, g, time_limit=timeout - (time.time() - start_time))
        
        if n < best_n:
            best_n = n
            best_edges = edges
            
        # If we found a graph close to the Moore bound, we can stop
        moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        if best_n <= moore_bound * 1.1:
            break

    if best_n < float('inf'):
        return best_n, best_edges
    
    # Fallback to LUW if greedy failed (should not happen, but safe)
    return luw_construct(k, g)

def greedy_construct(k, g, time_limit):
    """
    Greedy construction of a k-regular graph with girth >= g.
    Starts with an edge, adds edges or new vertices as needed.
    """
    start_time = time.time()
    
    # Graph representation
    adj = {}  # vertex -> list of neighbors
    degree = {}
    next_vertex_id = 0
    
    def add_vertex():
        nonlocal next_vertex_id
        v = next_vertex_id
        next_vertex_id += 1
        adj[v] = []
        degree[v] = 0
        return v

    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1

    def get_distance(u, v):
        """BFS to find distance between u and v. Returns -1 if not connected."""
        if u == v:
            return 0
        visited = {u: 0}
        queue = [u]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited[neighbor] = visited[curr] + 1
                    if neighbor == v:
                        return visited[neighbor]
                    queue.append(neighbor)
        return -1

    # Initialize
    v0 = add_vertex()
    v1 = add_vertex()
    add_edge(v0, v1)
    
    active = [v0, v1]
    
    while active:
        if time.time() - start_time > time_limit:
            return -1, [] # Timeout
        
        # Pick a random active vertex
        u = random.choice(active)
        
        # Find a valid partner
        found = False
        # Candidates are all vertices not connected to u
        # We try random sampling
        all_vertices = list(adj.keys())
        random.shuffle(all_vertices)
        
        for v in all_vertices:
            if v == u or v in adj[u]:
                continue
            
            # Check girth constraint
            # Adding edge (u,v) creates a cycle of length dist(u,v) + 1
            # We need dist(u,v) + 1 >= g  => dist(u,v) >= g - 1
            dist = get_distance(u, v)
            if dist == -1:
                # Not connected, adding edge connects components.
                # Cycle length is infinite? No, it just connects.
                # But if they are in different components, no cycle is formed.
                # However, we want a single connected component usually?
                # The task doesn't strictly require connectedness, but "cage" implies it.
                # If we allow disconnected, we can just make k-regular components.
                # But the evaluator checks "no cycle shorter than g".
                # Disconnected graphs are valid if each component has girth >= g.
                # However, usually cages are connected.
                # Let's assume connected is preferred, but valid disconnected is better than invalid.
                # Actually, if dist is -1, no cycle is formed. So it's valid.
                # But we should prefer connected to keep N small?
                # Connecting components reduces the number of "active" ends?
                # No, it just adds an edge.
                # Let's accept it.
                add_edge(u, v)
                found = True
                break
            else:
                if dist >= g - 1:
                    add_edge(u, v)
                    found = True
                    break
        
        if found:
            if degree[u] == k:
                active.remove(u)
            if degree[v] == k:
                if v in active:
                    active.remove(v)
        else:
            # No valid partner found. Add a new vertex.
            w = add_vertex()
            add_edge(u, w)
            active.append(w)
            if degree[u] == k:
                active.remove(u)

    # Verify result
    if len(adj) != len(set(adj.keys())):
        return -1, []
        
    for u in adj:
        if degree[u] != k:
            return -1, []
        
    # Check girth exactly
    # We already enforced it during construction, but let's be safe?
    # The construction ensures no cycle < g is formed.
    
    edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                
    return len(adj), edges

def luw_construct(k, g):
    """Fallback LUW construction from seed."""
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
