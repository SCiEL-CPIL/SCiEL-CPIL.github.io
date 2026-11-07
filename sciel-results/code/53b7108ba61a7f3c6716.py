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
                    if (u, v) not in used_edges and (v, u) not in used_edges:
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
        
        # Find a root (vertex 0 or any vertex)
        # Use DFS-based backtracking to assign host vertices
        def backtrack(assigned, pos):
            """
            assigned: list of length k, assigned[v] = host vertex or -1
            pos: next vertex to assign in the order
            """
            if pos == k:
                # Check all tree edges map to valid K20 edges and are unused
                for (a, b) in tree_edges:
                    ha, hb = assigned[a], assigned[b]
                    if ha > hb:
                        ha, hb = hb, ha
                    if (ha, hb) in used_edges:
                        return False
                # All good, commit
                for (a, b) in tree_edges:
                    ha, hb = assigned[a], assigned[b]
                    if ha > hb:
                        ha, hb = hb, ha
                    used_edges.add((ha, hb))
                result[idx] = list(assigned)
                return True
            return False
        
        # Use a smarter approach: build the tree embedding incrementally
        # Start with vertex 0, then extend
        host_assignment = [-1] * k
        host_assignment[0] = -1  # Will be set
        
        # Try all possible host vertices for the root
        success = False
        for root_host in range(n):
            host_assignment[0] = root_host
            # BFS/DFS to assign remaining vertices
            visited = [False] * k
            visited[0] = True
            queue = [0]
            valid = True
            while queue and valid:
                u = queue.pop(0)
                for v in adj[u]:
                    if not visited[v]:
                        # Try all available host vertices adjacent to host_assignment[u]
                        found = False
                        for h in range(n):
                            if h == host_assignment[u]:
                                continue
                            if h in host_assignment:
                                continue
                            # Check if (host_assignment[u], h) is an available edge
                            a, b = host_assignment[u], h
                            if a > b:
                                a, b = b, a
                            if (a, b) in used_edges:
                                continue
                            # Also check it's not already assigned to another vertex
                            if h in host_assignment:
                                continue
                            host_assignment[v] = h
                            visited[v] = True
                            queue.append(v)
                            found = True
                            break
                        if not found:
                            valid = False
                            break
            if valid:
                # Verify all edges are distinct and available
                ok = True
                for (a, b) in tree_edges:
                    ha, hb = host_assignment[a], host_assignment[b]
                    if ha > hb:
                        ha, hb = hb, ha
                    if (ha, hb) in used_edges:
                        ok = False
                        break
                if ok:
                    for (a, b) in tree_edges:
                        ha, hb = host_assignment[a], host_assignment[b]
                        if ha > hb:
                            ha, hb = hb, ha
                        used_edges.add((ha, hb))
                    result[idx] = list(host_assignment)
                    success = True
                    break
            # Reset for next root_host
            host_assignment = [-1] * k
        
        if not success:
            result[idx] = None
    
    return result
