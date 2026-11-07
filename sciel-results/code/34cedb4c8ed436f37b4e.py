import time
from random import Random
from collections import defaultdict

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
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    
    trees = instance_trees(20, 611)
    N = 20
    
    # Precompute tree structures
    tree_adj = []
    tree_degrees = []
    tree_edges = []
    for k in range(20):
        edges = trees[k]
        adj = defaultdict(list)
        deg = [0] * (k + 1)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_degrees.append(deg)
        tree_edges.append(edges)
    
    # Total edges in K20 = 190, sum of tree edges = 0+1+...+19 = 190
    # So we must use ALL edges exactly once.
    
    # Key insight: Use a fast backtracking that assigns tree vertices one by one
    # with strong forward checking. Process trees in order of decreasing size.
    
    # Represent edges as sorted tuples for fast lookup
    all_edges_set = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_edges_set.add((i, j))
    
    # For each tree, precompute a good vertex ordering (BFS from highest degree)
    tree_orders = []
    for k in range(20):
        if k == 0:
            tree_orders.append([0])
            continue
        deg = tree_degrees[k]
        root = max(range(k + 1), key=lambda v: deg[v])
        visited = [False] * (k + 1)
        order = [root]
        visited[root] = True
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in tree_adj[k][u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    queue.append(v)
        tree_orders.append(order)
    
    # Embedding order: largest trees first
    embed_order = list(range(19, -1, -1))
    
    # Fast solver using iterative backtracking with pruning
    # State: used_edges (set of (i,j) tuples), embeddings (list of lists)
    
    def solve_with_order(order):
        used_edges = set()
        embeddings = [None] * 20
        
        for tree_idx in order:
            k = tree_idx + 1
            edges = tree_edges[tree_idx]
            adj = tree_adj[tree_idx]
            deg = tree_degrees[tree_idx]
            
            if k == 1:
                # Pick any unused vertex (all vertices are "used" in the sense of being assigned)
                # Actually for k=1, we just need one vertex with no edges
                embeddings[tree_idx] = [0]
                continue
            
            # BFS order for this tree
            root = max(range(k), key=lambda v: deg[v])
            visited = [False] * k
            t_order = [root]
            visited[root] = True
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        t_order.append(v)
                        queue.append(v)
            
            # Backtracking: assign host vertices to tree vertices in t_order
            assign = {}  # tree_vertex -> host_vertex
            assigned_hosts = set()
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                
                if pos == k:
                    # All assigned, verify edges and commit
                    for u, v in edges:
                        a, b = assign[u], assign[v]
                        if a > b:
                            a, b = b, a
                        edge = (a, b)
                        if edge in used_edges:
                            return False
                    for u, v in edges:
                        a, b = assign[u], assign[v]
                        if a > b:
                            a, b = b, a
                        used_edges.add((a, b))
                    embeddings[tree_idx] = [assign[i] for i in range(k)]
                    return True
                
                tv = t_order[pos]
                # Find valid host vertices
                for h in range(N):
                    if h in assigned_hosts:
                        continue
                    
                    # Check against already-assigned neighbors
                    ok = True
                    for nb in adj[tv]:
                        if nb in assign:
                            a, b = h, assign[nb]
                            if a > b:
                                a, b = b, a
                            if (a, b) in used_edges:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    # Forward checking: can remaining unassigned tree vertices
                    # still be placed? (light check: just check if there are enough
                    # unused host vertices adjacent to needed edges)
                    assign[tv] = h
                    assigned_hosts.add(h)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    del assign[tv]
                    assigned_hosts.remove(h)
                
                return False
            
            if not backtrack(0):
                return None
        
        return embeddings
    
    # Try the main solve
    result = solve_with_order(embed_order)
    if result is not None:
        # Verify
        used = set()
        total = 0
        for i in range(20):
            if result[i] is None:
                return [None] * 20
            for u, v in tree_edges[i]:
                a, b = result[i][u], result[i][v]
                if a > b:
                    a, b = b, a
                if (a, b) in used:
                    return [None] * 20
                used.add((a, b))
                total += 1
        return result
    
    # If main failed, try randomized restarts
    rng = Random(12345)
    for attempt in range(5000):
        if time.time() > deadline - 3:
            break
        
        # Create a perturbed order
        order = embed_order[:]
        for _ in range(rng.randint(1, 5)):
            i = rng.randrange(len(order) - 1)
            order[i], order[i + 1] = order[i + 1], order[i]
        
        result = solve_with_order(order)
        if result is not None:
            used = set()
            total = 0
            for i in range(20):
                if result[i] is None:
                    return [None] * 20
                for u, v in tree_edges[i]:
                    a, b = result[i][u], result[i][v]
                    if a > b:
                        a, b = b, a
                    if (a, b) in used:
                        return [None] * 20
                    used.add((a, b))
                    total += 1
            return result
    
    return [None] * 20
