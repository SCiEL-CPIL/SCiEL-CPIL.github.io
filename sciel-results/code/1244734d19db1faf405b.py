from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, 611)
    n = 20
    
    # Track which K20 edges are used
    used_edges = set()
    
    # Result: list of 20 entries, entry k-1 is None or list of k host vertices
    result = [None] * 20
    
    # Process trees in order of decreasing size (harder ones first)
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    for idx in tree_indices:
        k = idx + 1  # Tree T_k has k vertices
        if k == 1:
            result[idx] = None
            continue
        if k == 2:
            # Find any available edge
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in used_edges:
                        used_edges.add((u, v))
                        result[idx] = [u, v]
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For larger trees, use backtracking to find an embedding
        tree_edges = trees[idx]
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (a, b) in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Compute degree in tree for ordering
        tree_deg = [len(adj[v]) for v in range(k)]
        
        # Use backtracking: assign host vertices one by one
        # Order vertices by degree (highest first) to constrain early
        # But we need to respect tree structure - use BFS order from a root
        # Pick root as vertex with max degree
        root = max(range(k), key=lambda v: tree_deg[v])
        
        # BFS order from root
        from collections import deque
        order = []
        visited = [False] * k
        visited[root] = True
        q = deque([root])
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        
        # Backtracking search
        host_assign = [-1] * k
        used_hosts = set()
        
        def search(pos):
            if pos == k:
                # Verify all tree edges map to unused K20 edges
                for (a, b) in tree_edges:
                    ha, hb = host_assign[a], host_assign[b]
                    e = (ha, hb) if ha < hb else (hb, ha)
                    if e in used_edges:
                        return False
                # Commit
                for (a, b) in tree_edges:
                    ha, hb = host_assign[a], host_assign[b]
                    e = (ha, hb) if ha < hb else (hb, ha)
                    used_edges.add(e)
                result[idx] = list(host_assign)
                return True
            
            u = order[pos]
            # Try all host vertices not yet used
            for h in range(n):
                if h in used_hosts:
                    continue
                # Check that for all already-assigned neighbors of u,
                # the edge (host_assign[neighbor], h) is not in used_edges
                valid = True
                for v in adj[u]:
                    if host_assign[v] != -1:
                        hv = host_assign[v]
                        e = (h, hv) if h < hv else (hv, h)
                        if e in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                host_assign[u] = h
                used_hosts.add(h)
                if search(pos + 1):
                    return True
                host_assign[u] = -1
                used_hosts.remove(h)
            
            return False
        
        # Try different roots if needed
        success = False
        # Try root first, then other high-degree vertices
        roots_to_try = sorted(range(k), key=lambda v: -tree_deg[v])
        for r in roots_to_try:
            # Rebuild BFS order from this root
            order = []
            visited = [False] * k
            visited[r] = True
            q = deque([r])
            while q:
                u = q.popleft()
                order.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            
            host_assign = [-1] * k
            used_hosts = set()
            
            # Redefine search with this order
            def search2(pos):
                if pos == k:
                    for (a, b) in tree_edges:
                        ha, hb = host_assign[a], host_assign[b]
                        e = (ha, hb) if ha < hb else (hb, ha)
                        if e in used_edges:
                            return False
                    for (a, b) in tree_edges:
                        ha, hb = host_assign[a], host_assign[b]
                        e = (ha, hb) if ha < hb else (hb, ha)
                        used_edges.add(e)
                    result[idx] = list(host_assign)
                    return True
                
                u = order[pos]
                for h in range(n):
                    if h in used_hosts:
                        continue
                    valid = True
                    for v in adj[u]:
                        if host_assign[v] != -1:
                            hv = host_assign[v]
                            e = (h, hv) if h < hv else (hv, h)
                            if e in used_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    host_assign[u] = h
                    used_hosts.add(h)
                    if search2(pos + 1):
                        return True
                    host_assign[u] = -1
                    used_hosts.remove(h)
                
                return False
            
            if search2(0):
                success = True
                break
        
        if not success:
            result[idx] = None
    
    return result
