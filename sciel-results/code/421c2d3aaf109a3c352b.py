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
        
        # For k >= 2, use backtracking
        tree_edges = trees[idx]
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (a, b) in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
            
        # Determine an order of vertices to assign (BFS order from 0)
        order = []
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    
        assignment = [-1] * k
        
        def dfs(pos):
            if pos == k:
                # All vertices assigned. Verify edges are unused.
                for (a, b) in tree_edges:
                    ha, hb = assignment[a], assignment[b]
                    key = (min(ha, hb), max(ha, hb))
                    if key in used_edges:
                        return False
                # Commit
                for (a, b) in tree_edges:
                    ha, hb = assignment[a], assignment[b]
                    used_edges.add((min(ha, hb), max(ha, hb)))
                result[idx] = list(assignment)
                return True
            
            v = order[pos]
            # Try all possible host vertices for v
            for h in range(n):
                # Check if h is already used by another tree vertex
                if h in assignment:
                    continue
                
                # Check constraints with already assigned neighbors
                valid = True
                for u in adj[v]:
                    if assignment[u] != -1:
                        # Neighbor u is already assigned
                        key = (min(assignment[u], h), max(assignment[u], h))
                        if key in used_edges:
                            valid = False
                            break
                
                if valid:
                    assignment[v] = h
                    if dfs(pos + 1):
                        return True
                    assignment[v] = -1
                    
            return False
            
        if not dfs(0):
            result[idx] = None
            
    return result
