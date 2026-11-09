"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    # used_edges[v] = set of vertices u such that edge {u, v} is used
    used_edges = [set() for _ in range(n)]
    
    # Precompute adjacency for each tree
    tree_adj = []
    for edges in trees:
        adj = [set() for _ in range(len(edges) + 1)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        tree_adj.append(adj)
    
    result = [None] * n
    
    for k in range(n):
        if k == 0:
            result[k] = None
            continue
        if k == 1:
            # T2 is a single edge (0,1). Find any unused edge.
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in used_edges[u]:
                        result[k] = [u, v]
                        used_edges[u].add(v)
                        used_edges[v].add(u)
                        found = True
                        break
                if found:
                    break
            if not found:
                result[k] = None
            continue
        
        # For tree k (0-indexed), we have k+1 vertices
        num_vertices = k + 1
        adj = tree_adj[k]
        
        # Backtracking search
        assignment = [-1] * num_vertices
        available = list(range(n))
        
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(num_vertices), key=lambda v: len(adj[v]), reverse=True)
        
        def backtrack(idx):
            if idx == num_vertices:
                return True
            
            v = order[idx]
            
            # Try assigning each available host vertex
            for i in range(len(available)):
                host = available[i]
                
                # Check if all already-assigned neighbors are compatible
                ok = True
                for u in adj[v]:
                    if assignment[u] != -1:
                        # u is already assigned to assignment[u]
                        # The edge {host, assignment[u]} must not be used
                        if assignment[u] in used_edges[host]:
                            ok = False
                            break
                        # Also check if host is in used_edges[assignment[u]] (symmetric)
                        # But we maintain both, so one check suffices
                if not ok:
                    continue
                
                # Check that host is not already used by another vertex in this tree
                # (shouldn't happen since we remove from available, but let's be safe)
                # Actually, we need to ensure no two vertices of this tree map to same host.
                # Since we pick from 'available' and remove, this is guaranteed.
                
                # Assign
                assignment[v] = host
                available[i] = available[-1]
                available.pop()
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                available.append(host)
                available[i], available[-1] = available[-1], available[i]
                assignment[v] = -1
            
            return False
        
        if backtrack(0):
            # Build result list: position i is image of vertex i
            result[k] = [assignment[i] for i in range(num_vertices)]
            # Mark edges as used
            for u, v in trees[k]:
                hu = assignment[u]
                hv = assignment[v]
                used_edges[hu].add(hv)
                used_edges[hv].add(hu)
        else:
            result[k] = None
    
    # Convert to list of ints
    for i in range(n):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result
# EVOLVE-BLOCK-END
