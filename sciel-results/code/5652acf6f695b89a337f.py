"""Combined program: structural backtracking from inspiration, refined with deadline guards and heuristics."""
import time
import random
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n = 20
    trees = instance_trees(n, 611)
    
    # All possible edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Pre-compute tree structures: adjacency lists and edge lists
    tree_adjs = []
    tree_edge_counts = []
    for t_idx in range(n):
        k = t_idx + 1
        if k == 1:
            tree_adjs.append([[]])
            tree_edge_counts.append(0)
            continue
        adj = [[] for _ in range(k)]
        for u, v in trees[t_idx]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)
        tree_edge_counts.append(len(trees[t_idx]))
    
    best_value = 0
    best_solution = [None] * n
    
    # Pack trees in order of decreasing edge count (largest first)
    # T20 (19 edges), T19 (18 edges), ..., T2 (1 edge), T1 (0 edges)
    pack_order = list(range(n))
    pack_order.sort(key=lambda x: tree_edge_counts[x], reverse=True)
    
    # We'll do multiple random restarts
    max_restarts = 5000
    restart = 0
    
    while restart < max_restarts:
        if time.time() > deadline:
            break
        
        restart += 1
        available = set(all_edges)
        current_solution = [None] * n
        success = True
        
        # Process trees in pack_order
        for t_idx in pack_order:
            if time.time() > deadline:
                success = False
                break
            
            k = t_idx + 1
            if k == 1:
                # T1: No edges, just pick a random vertex
                v = random.randint(0, n - 1)
                current_solution[t_idx] = [v]
                continue
            
            # Try to embed tree t_idx using backtracking
            adj = tree_adjs[t_idx]
            
            # Find a root: pick a vertex with max degree for better constraint propagation
            degrees = [len(adj[i]) for i in range(k)]
            root = max(range(k), key=lambda i: degrees[i])
            
            # DFS with backtracking to find a valid embedding
            # mapping[i] = graph vertex for tree vertex i, -1 if unassigned
            mapping = [-1] * k
            used_vertices = set()
            
            # Order children by degree (descending) for better pruning
            # Pre-compute child order for each node (excluding parent)
            def dfs(node, parent):
                if time.time() > deadline:
                    return False
                
                # Determine candidates for this node
                if parent != -1:
                    parent_gv = mapping[parent]
                    # Candidates must be adjacent to parent_gv in available edges
                    candidates = []
                    for gv in range(n):
                        if gv in used_vertices:
                            continue
                        e = (min(parent_gv, gv), max(parent_gv, gv))
                        if e in available:
                            candidates.append(gv)
                else:
                    candidates = list(range(n))
                
                # Shuffle for randomization
                random.shuffle(candidates)
                
                for gv in candidates:
                    if gv in used_vertices:
                        continue
                    
                    # Check edge constraint if parent exists
                    if parent != -1:
                        p_gv = mapping[parent]
                        e = (min(p_gv, gv), max(p_gv, gv))
                        if e not in available:
                            continue
                    
                    # Assign
                    mapping[node] = gv
                    used_vertices.add(gv)
                    
                    # Recurse to children
                    all_ok = True
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if not dfs(neighbor, node):
                            all_ok = False
                            break
                    
                    if all_ok:
                        return True
                    
                    # Undo
                    used_vertices.remove(gv)
                    mapping[node] = -1
                
                return False
            
            if not dfs(root, -1):
                success = False
                break
        
        if success:
            # Calculate total edges
            total_edges = sum(len(trees[t_idx]) for t_idx in range(n) if current_solution[t_idx] is not None)
            if total_edges > best_value:
                best_value = total_edges
                best_solution = [list(current_solution[t_idx]) if current_solution[t_idx] is not None else None for t_idx in range(n)]
            
            # If we packed all trees, we're done
            if best_value == 190:
                break
        
        # If we didn't succeed, try another restart
        # To add diversity, we can slightly randomize the pack order among trees of similar size
        if restart % 100 == 0:
            # Shuffle the order within groups of similar size
            pass
    
    # Ensure return format: list of 20 entries, each None or list of distinct ints
    result = []
    for t_idx in range(n):
        if best_solution[t_idx] is None:
            result.append(None)
        else:
            # Ensure it's a list of Python ints
            result.append([int(v) for v in best_solution[t_idx]])
    
    return result
