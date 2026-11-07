import time
import random
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
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency for each tree
    tree_adj = []
    for k in range(1, n + 1):
        edges = trees[k - 1]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Precompute parent and order for each tree (rooted at 0)
    tree_parent = []
    tree_order = []
    for k in range(1, n + 1):
        if k == 1:
            tree_parent.append([-1])
            tree_order.append([0])
            continue
        adj = tree_adj[k - 1]
        parent = [-1] * k
        parent[0] = -2  # root
        order_list = [0]
        for v in order_list:
            for nb in adj[v]:
                if parent[nb] == -1:
                    parent[nb] = v
                    order_list.append(nb)
        tree_parent.append(parent)
        tree_order.append(order_list)
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Process trees in order of decreasing size (most constrained first)
    # Tree k has k-1 edges, so order is k=20, 19, ..., 2, 1
    order = list(range(n - 1, -1, -1))  # indices 19 down to 0
    
    best_result = [None] * n
    best_edges = 0
    
    def try_solution():
        nonlocal best_result, best_edges
        used_edges = set()
        embedding = [None] * n
        
        for idx in order:
            k = idx + 1  # tree size
            tree_edges = trees[idx]
            
            if k == 1:
                # T1: no edges, just pick a vertex
                embedding[idx] = [random.randint(0, n - 1)]
                continue
            
            # Try to find an embedding of this tree
            # Use DFS with backtracking
            adj = tree_adj[idx]
            parent = tree_parent[idx]
            order_list = tree_order[idx]
            
            mapping = [-1] * k
            
            def dfs(pos):
                if time.time() > deadline:
                    return False
                
                if pos == len(order_list):
                    # All vertices assigned
                    # Verify all edges are available
                    for u, v in tree_edges:
                        gu, gv = mapping[u], mapping[v]
                        e = (min(gu, gv), max(gu, gv))
                        if e in used_edges:
                            return False
                    return True
                
                tv = order_list[pos]
                
                # Determine candidates
                if parent[tv] != -2:  # not root
                    p_tv = parent[tv]
                    p_gv = mapping[p_tv]
                    # Candidates must be adjacent to p_gv via an available edge
                    candidates = [v for v in range(n) 
                                  if v != p_gv 
                                  and (min(p_gv, v), max(p_gv, v)) not in used_edges]
                else:
                    candidates = list(range(n))
                
                random.shuffle(candidates)
                
                for gv in candidates:
                    # Check if gv is already used by another tree vertex
                    if gv in mapping:
                        continue
                    
                    # Check edges to already-assigned neighbors
                    ok = True
                    for nb in adj[tv]:
                        if nb == parent[tv]:
                            continue
                        if mapping[nb] != -1:
                            # nb is already assigned
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
            
            # Try multiple random roots to increase diversity
            found = False
            for root_attempt in range(5):
                if time.time() > deadline:
                    return False
                
                # Reset mapping
                mapping = [-1] * k
                
                # Re-root if needed (for simplicity, always use root 0)
                # But we can try different root orders by shuffling the order_list
                # For now, just try the standard DFS with random candidate ordering
                
                if dfs(0):
                    found = True
                    break
            
            if not found:
                return False
            
            # Commit this embedding
            for u, v in tree_edges:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                used_edges.add(e)
            
            embedding[idx] = mapping[:]
        
        # All trees embedded successfully
        total_edges = len(used_edges)
        if total_edges > best_edges:
            best_edges = total_edges
            best_result = [list(e) if e is not None else None for e in embedding]
        return True
    
    # Try multiple random restarts with different seeds
    restart = 0
    while time.time() < deadline:
        restart += 1
        random.seed(611 + restart)
        try_solution()
        if best_edges == 190:  # Maximum possible
            break
    
    # Return best result found
    if best_edges > 0:
        return best_result
    
    # Fallback: just T20 identity
    return [None] * 19 + [list(range(20))]
