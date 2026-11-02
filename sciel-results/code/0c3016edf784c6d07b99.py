import time
import random
import heapq

def _compute_girth(adj, n, max_depth=10):
    """
    Compute the girth of the graph using BFS from each vertex.
    Returns the shortest cycle length, or infinity if none found up to max_depth.
    """
    min_cycle = max_depth + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    return min_cycle

def _random_k_regular(n, k, seed=None):
    """
    Generate a random simple k-regular graph using the configuration model
    followed by 2-switches to remove multi-edges and self-loops.
    """
    if seed is not None:
        random.seed(seed)
    
    # Configuration model
    half_edges = list(range(n * k))
    random.shuffle(half_edges)
    
    # Pair up half-edges
    adj = [set() for _ in range(n)]
    for i in range(n * k // 2):
        u = half_edges[2 * i] // k
        v = half_edges[2 * i + 1] // k
        if u == v:
            # Self-loop, try to fix later or restart
            return None
        if v in adj[u]:
            # Multi-edge, try to fix later or restart
            return None
        adj[u].add(v)
        adj[v].add(u)
    
    # Check degrees
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    # Remove self-loops and multi-edges via 2-switches
    # This is a simplified version; a full implementation would be more robust
    # For now, we just return None if any issues were detected in the initial pairing
    # A more robust approach would be to iteratively fix issues
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    return adj

def _edge_switch_improve(adj, n, k, g, time_limit=50.0):
    """
    Try to increase the girth of the graph by performing edge switches.
    A switch replaces edges (u,v) and (w,x) with (u,w) and (v,x) if it
    increases the girth or maintains it at >= g.
    """
    start_time = time.time()
    current_girth = _compute_girth(adj, n, max_depth=g)
    
    if current_girth >= g:
        return adj, current_girth
    
    # Build a list of all edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    # Map edge to index
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # For each edge, find potential switch partners
    # This is expensive, so we'll sample
    while time.time() - start_time < time_limit:
        # Pick a random edge (u, v)
        u, v = random.choice(edges)
        
        # Pick a random other edge (w, x) not sharing vertices with (u, v)
        # and not adjacent in a way that would create self-loops
        attempts = 0
        while attempts < 100:
            w, x = random.choice(edges)
            if w == u or w == v or x == u or x == v:
                attempts += 1
                continue
            if w == x:
                attempts += 1
                continue
            if (u, w) in edge_index or (v, w) in edge_index or (u, x) in edge_index or (v, x) in edge_index:
                # This would create a multi-edge
                attempts += 1
                continue
            
            # Check if the switch increases girth
            # Temporarily apply switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
            
            new_girth = _compute_girth(adj, n, max_depth=g)
            
            # Revert switch
            adj[u].remove(w)
            adj[w].remove(u)
            adj[v].remove(x)
            adj[x].remove(v)
            
            adj[u].add(v)
            adj[v].add(u)
            adj[w].add(x)
            adj[x].add(w)
            
            if new_girth > current_girth:
                # Apply switch for real
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                
                # Update edge list
                edges.remove((u, v))
                edges.remove((w, x))
                edges.append((u, w))
                edges.append((v, x))
                edge_index = {e: i for i, e in enumerate(edges)}
                
                current_girth = new_girth
                
                if current_girth >= g:
                    return adj, current_girth
            
            attempts += 1
        
        # If we couldn't make progress in this batch, break
        break
    
    return adj, current_girth

def _construct_by_switching(k, g, n, time_limit=50.0):
    """
    Construct a k-regular graph of girth >= g on n vertices by starting
    with a random k-regular graph and improving its girth via edge switches.
    """
    start_time = time.time()
    
    # Try multiple random seeds
    for seed in range(100):
        if time.time() - start_time > time_limit:
            break
        
        adj = _random_k_regular(n, k, seed=seed)
        if adj is None:
            continue
        
        adj, girth = _edge_switch_improve(adj, n, k, g, time_limit=min(10.0, time_limit - (time.time() - start_time)))
        
        if girth >= g:
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
    
    return None

def _luw_construct(k, g):
    # Copy of the seed logic
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
    # For the specific case k=5, g=9, try edge switching
    if k == 5 and g == 9:
        # Try a few sizes
        # The best known is 1116.
        # We try slightly larger to increase probability of success.
        target_ns = [1116, 1120, 1150, 1200, 1500]
        
        # Time budget: 100s for switching, 10s for fallback
        total_time = 100.0
        start_time = time.time()
        
        for n in target_ns:
            remaining = total_time - (time.time() - start_time)
            if remaining < 5.0:
                break
            
            res = _construct_by_switching(k, g, n, time_limit=remaining)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
