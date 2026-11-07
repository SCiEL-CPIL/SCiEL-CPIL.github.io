import time
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
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency, parent, order for each tree
    tree_adj = []
    tree_parent = []
    tree_order = []
    
    for k in range(1, n + 1):
        if k == 1:
            tree_adj.append([[]])
            tree_parent.append([-1])
            tree_order.append([0])
            continue
        
        edges = trees[k - 1]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        
        # Root at 0, compute parent and BFS order
        parent = [-1] * k
        parent[0] = -2
        order_list = [0]
        for v in order_list:
            for nb in adj[v]:
                if parent[nb] == -1:
                    parent[nb] = v
                    order_list.append(nb)
        tree_parent.append(parent)
        tree_order.append(order_list)
    
    # Order: process trees by increasing size (least constrained first)
    # Then larger trees have more freedom
    # Actually, let's try a different strategy: process in decreasing size
    # so the most constrained (largest) trees get first pick of edges
    # But actually, small trees are easier to place, so placing them first
    # might work better. Let's try both and see.
    
    # Let's use a greedy + backtracking hybrid with multiple restarts
    # Key insight: process trees in order that maximizes flexibility
    
    # Strategy 1: Process in decreasing size (T20 first, T1 last)
    # Strategy 2: Process in increasing size (T1 first, T20 last)
    # Strategy 3: Process in a heuristic order based on tree structure
    
    # For each tree, we'll use a DFS with backtracking to find an embedding
    # that doesn't conflict with already-used edges
    
    best_result = [None] * n
    best_edges = 0
    
    def try_solution(rng, order):
        nonlocal best_result, best_edges
        
        used_edges = set()
        embedding = [None] * n
        
        for idx in order:
            k = idx + 1  # tree size (T_k has k vertices)
            
            if k == 1:
                # Single vertex: just pick any unused vertex... actually vertices can be reused
                # The constraint is only on edges
                embedding[idx] = [rng.randint(0, n - 1)]
                continue
            
            if k == 2:
                # Two vertices with one edge
                found = False
                for _ in range(100):
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v:
                        continue
                    e = (min(u, v), max(u, v))
                    if e not in used_edges:
                        embedding[idx] = [u, v]
                        used_edges.add(e)
                        found = True
                        break
                if not found:
                    # Try all pairs
                    for u in range(n):
                        for v in range(u + 1, n):
                            e = (u, v)
                            if e not in used_edges:
                                embedding[idx] = [u, v]
                                used_edges.add(e)
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return False
                continue
            
            # For trees with k >= 3, use DFS backtracking
            adj = tree_adj[idx]
            parent = tree_parent[idx]
            order_list = tree_order[idx]
            
            mapping = [-1] * k
            mapping[0] = rng.randint(0, n - 1)
            
            def dfs(pos):
                if time.time() > deadline:
                    return False
                
                if pos == len(order_list):
                    # Verify all edges are valid and unused
                    for u, v in trees[idx]:
                        gu, gv = mapping[u], mapping[v]
                        e = (min(gu, gv), max(gu, gv))
                        if e in used_edges:
                            return False
                    return True
                
                tv = order_list[pos]
                
                if parent[tv] != -2:
                    p_tv = parent[tv]
                    p_gv = mapping[p_tv]
                    # Candidates: all vertices except p_gv, and edge (p_gv, v) must be unused
                    candidates = []
                    for v in range(n):
                        if v == p_gv:
                            continue
                        e = (min(p_gv, v), max(p_gv, v))
                        if e not in used_edges:
                            # Also check that v is not already used by another vertex in this tree
                            if v not in mapping:
                                candidates.append(v)
                else:
                    candidates = [v for v in range(n) if v not in mapping]
                
                # Shuffle candidates for randomization
                rng.shuffle(candidates)
                
                # Heuristic: try candidates that have more available neighbors first
                # Actually, let's just try in shuffled order
                
                for gv in candidates:
                    # Check if any already-mapped children would conflict
                    ok = True
                    for nb in adj[tv]:
                        if nb == parent[tv]:
                            continue
                        if mapping[nb] != -1:
                            e = (min(gv, mapping[nb]), max(gv, mapping[nb]))
                            if e in used_edges:
                                ok = False
                                break
                    
                    if not ok:
                        continue
                    
                    mapping[tv] = gv
                    if dfs(pos + 1):
                        return True
                    mapping[tv] = -1
                
                return False
            
            found = False
            # Try multiple root placements
            for _ in range(20):
                if time.time() > deadline:
                    return False
                
                mapping = [-1] * k
                mapping[0] = rng.randint(0, n - 1)
                
                if dfs(0):
                    found = True
                    break
            
            if not found:
                return False
            
            # Mark edges as used
            for u, v in trees[idx]:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                used_edges.add(e)
            
            embedding[idx] = mapping[:]
        
        total_edges = len(used_edges)
        if total_edges > best_edges:
            best_edges = total_edges
            best_result = [list(e) if e is not None else None for e in embedding]
        return True
    
    # Try different processing orders
    # Order 1: increasing size (T1, T2, ..., T20)
    order_increasing = list(range(n))
    # Order 2: decreasing size (T20, T19, ..., T1)
    order_decreasing = list(range(n - 1, -1, -1))
    # Order 3: random
    order_random = list(range(n))
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        rng = Random(611 + restart * 7919)
        
        # Alternate between different orders
        if restart % 3 == 1:
            order = order_increasing
        elif restart % 3 == 2:
            order = order_decreasing
        else:
            rng.shuffle(order_random)
            order = order_random[:]
        
        try_solution(rng, order)
        
        if best_edges == 190:
            break
    
    if best_edges > 0:
        return best_result
    
    # Fallback: just embed T20
    return [None] * 19 + [list(range(20))]
