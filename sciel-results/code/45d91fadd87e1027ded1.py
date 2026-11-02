<<<<<<< SEARCH
def _construct_random_regular(n, k, seed):
    """Attempt to construct a random k-regular graph on n vertices using configuration model."""
    rng = random.Random(seed)
    
    # Configuration model: create n*k stubs
    stubs = list(range(n * k))
    # Assign stubs to vertices: vertex i gets stubs [i*k, (i+1)*k)
    # Shuffle and pair up
    rng.shuffle(stubs)
    
    # Create edges by pairing consecutive stubs
    adj = [[] for _ in range(n)]
    edge_set = set()
    valid = True
    
    for i in range(0, n * k, 2):
        u = stubs[i] // k
        v = stubs[i + 1] // k
        if u == v:
            valid = False
            break
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            valid = False
            break
        edge_set.add((a, b))
        adj[u].append(v)
        adj[v].append(u)
    
    if not valid:
        return None
    
    # Check regularity (should be automatic from construction)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    edges = list(edge_set)
    return n, edges
=======
def _construct_random_regular(n, k, seed):
    """Attempt to construct a random k-regular graph on n vertices using configuration model."""
    rng = random.Random(seed)
    
    # Configuration model: create n*k stubs
    stubs = list(range(n * k))
    # Assign stubs to vertices: vertex i gets stubs [i*k, (i+1)*k)
    # Shuffle and pair up
    rng.shuffle(stubs)
    
    # Create edges by pairing consecutive stubs
    adj = [[] for _ in range(n)]
    edge_set = set()
    valid = True
    
    for i in range(0, n * k, 2):
        u = stubs[i] // k
        v = stubs[i + 1] // k
        if u == v:
            valid = False
            break
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            valid = False
            break
        edge_set.add((a, b))
        adj[u].append(v)
        adj[v].append(u)
    
    if not valid:
        return None
    
    # Check regularity (should be automatic from construction)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    edges = list(edge_set)
    return n, edges


def _try_greedy_girth_search(k, g, target_n, deadline, max_attempts=50):
    """
    Try to construct a k-regular graph with girth >= g on target_n vertices
    using a greedy approach that avoids creating short cycles.
    """
    if time.time() > deadline - 10:
        return None
    
    best_result = None
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        result = _greedy_construct_with_girth(k, g, target_n, seed=attempt)
        if result is not None:
            n_try, edges_try = result
            if _verify_girth(n_try, edges_try, k, g):
                best_result = result
                break
    
    return best_result


def _greedy_construct_with_girth(k, g, target_n, seed=0):
    """
    Greedy construction: build graph vertex by vertex, choosing edges
    that don't create cycles shorter than g.
    """
    rng = random.Random(seed)
    
    # Start with a tree-like structure, then add edges to make it regular
    # This is heuristic and may not always succeed
    
    n = target_n
    if n < 2 * k:
        return None
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    # First, create a spanning tree using random DFS
    visited = [False] * n
    stack = [0]
    visited[0] = True
    parent = [-1] * n
    
    while stack:
        u = stack.pop()
        neighbors = list(range(n))
        rng.shuffle(neighbors)
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                adj[u].append(v)
                adj[v].append(u)
                edge_set.add((min(u, v), max(u, v)))
                stack.append(v)
                break
    
    # Now we have a tree. We need to add n*(k-1) more edges to make it k-regular
    # Each vertex currently has degree 1 (except root has 1, others have 1)
    # Actually in a tree, sum of degrees = 2*(n-1), so average degree < 2
    # We need to add edges to reach degree k
    
    # Track current degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # We need to add (n*k - 2*(n-1)) / 2 edges
    edges_to_add = (n * k - 2 * (n - 1)) // 2
    
    # Try to add edges that don't create short cycles
    # For simplicity, just try random pairs and check girth constraint
    
    added = 0
    attempts = 0
    max_attempts = 10000
    
    while added < edges_to_add and attempts < max_attempts:
        attempts += 1
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        
        if u == v:
            continue
        if degrees[u] >= k or degrees[v] >= k:
            continue
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            continue
        
        # Check if adding this edge creates a cycle shorter than g
        # This requires checking if there's a path from u to v of length < g-1
        # that doesn't use the edge (u,v)
        if _would_create_short_cycle(adj, u, v, g):
            continue
        
        # Add the edge
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((a, b))
        degrees[u] += 1
        degrees[v] += 1
        added += 1
    
    # Check if we achieved k-regularity
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    edges = list(edge_set)
    return n, edges


def _would_create_short_cycle(adj, u, v, g):
    """Check if adding edge (u,v) would create a cycle of length < g."""
    # BFS from u to find shortest path to v, not using direct edge
    if v in adj[u]:
        # Already adjacent, adding parallel edge not allowed (simple graph)
        return True
    
    visited = {u: 0}
    queue = deque([u])
    
    while queue:
        node, dist = queue.popleft()
        if dist >= g - 1:
            break
        for neighbor in adj[node]:
            if neighbor == v:
                # Found path of length dist + 1, cycle would be dist + 2
                if dist + 2 < g:
                    return True
                continue
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return False
>>>>>>> REPLACE

