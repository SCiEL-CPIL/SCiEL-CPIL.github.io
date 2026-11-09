import time

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with vertex ordering."""
    color = [-1] * n
    order = sorted(range(n), key=lambda v: sum(adj[v]), reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        for c in range(3):
            valid = True
            for u in range(n):
                if adj[v][u] and color[u] == c:
                    valid = False
                    break
            if valid:
                color[v] = c
                if backtrack(idx+1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 140
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    m = len(all_edges)
    
    # Clebsch graph as initial solution
    clebsch = set()
    for u in range(n):
        for v in range(u+1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                clebsch.add((u, v))
    
    best = set(clebsch)
    best_count = len(best)
    
    def can_add(edge_set, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        for w in range(n):
            if w == u or w == v:
                continue
            uw = (min(u, w), max(u, w))
            vw = (min(v, w), max(v, w))
            if uw in edge_set and vw in edge_set:
                return False
        return True
    
    # Greedy: try to add edges to current best
    current = set(best)
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for u, v in all_edges:
            if (u, v) in current:
                continue
            if can_add(current, u, v):
                current.add((u, v))
                changed = True
    
    # Check 3-colorability
    adj = [[False]*n for _ in range(n)]
    for u, v in current:
        adj[u][v] = adj[v][u] = True
    if not _is_3_colorable(adj, n) and len(current) > best_count:
        best = set(current)
        best_count = len(current)
    
    # Random restarts: remove some edges, re-optimize
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        
        current = set(best)
        # Remove a few random edges to escape local optimum
        if current:
            to_remove = random.sample(list(current), min(3, len(current)))
            for e in to_remove:
                current.discard(e)
        
        # Greedy re-add
        changed = True
        while changed and time.time() < deadline - 20:
            changed = False
            for u, v in all_edges:
                if (u, v) in current:
                    continue
                if can_add(current, u, v):
                    current.add((u, v))
                    changed = True
        
        # Check 3-colorability
        adj = [[False]*n for _ in range(n)]
        for u, v in current:
            adj[u][v] = adj[v][u] = True
        if not _is_3_colorable(adj, n) and len(current) > best_count:
            best = set(current)
            best_count = len(current)
    
    edges = [[u, v] for u, v in best]
    return edges
