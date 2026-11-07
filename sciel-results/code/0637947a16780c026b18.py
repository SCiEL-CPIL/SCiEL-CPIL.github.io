import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    trees = instance_trees(20, 611)
    results = [None] * 20
    used_edges = set()
    
    for k in range(20, 0, -1):
        if time.time() - start_time > timeout:
            break
            
        tree_edges = trees[k-1]
        if not tree_edges:
            results[k-1] = [0]
            continue
            
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        root = max(range(k), key=lambda x: len(adj[x]))
        
        parent = [-1] * k
        order = []
        stack = [root]
        parent[root] = root
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    stack.append(v)
                    
        assigned = [-1] * k
        used_host_vertices = set()
        
        # We will try multiple times with different randomizations if it fails
        # But first, a standard deterministic attempt
        attempts = 0
        max_attempts = 5
        found = False
        rng_local = Random(42)
        
        while not found and attempts < max_attempts:
            attempts += 1
            assigned = [-1] * k
            used_host_vertices = set()
            
            # Shuffle host vertices for root and children if attempts > 1
            host_order = list(range(20))
            if attempts > 1:
                rng_local.shuffle(host_order)
            
            def backtrack(idx):
                nonlocal found
                if idx == len(order):
                    found = True
                    return True
                
                u = order[idx]
                p = parent[u]
                
                # If we have a parent, the edge to parent must be valid
                # The host vertex for parent is fixed
                h_p = assigned[p] if p != u else None
                
                for h_u in host_order:
                    if h_u in used_host_vertices:
                        continue
                    
                    if h_p is not None:
                        edge = (min(h_p, h_u), max(h_p, h_u))
                        if edge in used_edges:
                            continue
                    
                    # Assign
                    assigned[u] = h_u
                    used_host_vertices.add(h_u)
                    if h_p is not None:
                        edge = (min(h_p, h_u), max(h_p, h_u))
                        used_edges.add(edge)
                        
                    if backtrack(idx + 1):
                        return True
                        
                    # Undo
                    assigned[u] = -1
                    used_host_vertices.remove(h_u)
                    if h_p is not None:
                        used_edges.remove(edge)
                        
                return False
            
            # Root assignment
            for h_root in host_order:
                assigned[root] = h_root
                used_host_vertices.add(h_root)
                
                if backtrack(1):
                    found = True
                    break
                    
                # Undo root
                assigned[root] = -1
                used_host_vertices.remove(h_root)
                
            if found:
                break
        
        if found:
            embedding = [0] * k
            for i in range(k):
                embedding[i] = assigned[i]
            results[k-1] = embedding
        else:
            results[k-1] = None

    return results
