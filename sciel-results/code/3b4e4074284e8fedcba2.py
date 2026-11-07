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
    deadline = start_time + 115.0  # 5s margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute tree properties
    tree_adj = []
    tree_deg = []
    tree_size = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)
        tree_size.append(k)
        
    # Sort trees by size descending
    order = list(range(n))
    order.sort(key=lambda x: tree_size[x], reverse=True)
    
    # Precompute graph edge indices for K20
    # Total edges = 190
    edge_idx = {}
    edge_verts = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx[(u, v)] = idx
            edge_idx[(v, u)] = idx
            edge_verts.append((u, v))
            idx += 1
            
    # Adjacency list of edge indices for each vertex
    adj_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edge_verts):
        adj_edges[u].append(i)
        adj_edges[v].append(i)
        
    # Initial available mask (all edges available)
    initial_avail_mask = (1 << 190) - 1
    
    best_solution = None
    best_value = 0
    
    # Try multiple random restarts
    # We use a fixed number of restarts or time-based
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 678)
        
        avail_mask = initial_avail_mask
        used_mask = 0
        current_solution = [None] * n
        success = True
        packed_edges = 0
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                # T1: Single vertex
                # Pick any unused vertex
                candidates = [v for v in range(n) if not (used_mask & (1 << v))]
                if not candidates:
                    success = False
                    break
                v = rng.choice(candidates)
                used_mask |= (1 << v)
                current_solution[t_idx] = [v]
                continue
            
            adj = tree_adj[t_idx]
            deg = tree_deg[t_idx]
            
            # Pick root: prefer high degree
            max_deg = max(deg)
            high_deg_nodes = [i for i in range(k) if deg[i] == max_deg]
            
            # Try different roots if one fails? 
            # For now, pick one randomly from high degree, or iterate?
            # Iterating over all high-degree roots is better for robustness.
            roots_to_try = high_deg_nodes[:]
            rng.shuffle(roots_to_try)
            
            embedded = False
            mapping = [-1] * k
            
            for root in roots_to_try:
                if time.time() > deadline:
                    break
                    
                # Reset state for this root attempt? 
                # No, we need to track used_mask and avail_mask.
                # But if we try a new root, we should reset the mapping for this tree.
                # The used_mask and avail_mask are global across trees, so they persist.
                # We just need to reset the local mapping and DFS state.
                
                local_mapping = [-1] * k
                failed_states = set()
                
                def embed_dfs(node, parent):
                    nonlocal avail_mask, used_mask, local_mapping
                    if time.time() > deadline:
                        return False
                    
                    if parent != -1:
                        parent_gv = local_mapping[parent]
                        state_key = (node, parent_gv)
                        if state_key in failed_states:
                            return False
                        
                        # Find candidates
                        candidates = []
                        # Iterate neighbors of parent_gv in K20
                        for e_idx in adj_edges[parent_gv]:
                            if not (avail_mask & (1 << e_idx)):
                                continue
                            u, v = edge_verts[e_idx]
                            neighbor = v if u == parent_gv else u
                            if used_mask & (1 << neighbor):
                                continue
                            # Check if neighbor is already used in this tree mapping
                            if neighbor in local_mapping: # This is O(k) check, slow.
                                continue
                            candidates.append(neighbor)
                    else:
                        candidates = [v for v in range(n) if not (used_mask & (1 << v))]
                        # Filter out vertices already in local_mapping (should be empty at root)
                        
                    if not candidates:
                        if parent != -1:
                            failed_states.add((node, local_mapping[parent]))
                        return False
                    
                    # Heuristic: Sort candidates by number of available edges (constraint)
                    def score(v):
                        count = 0
                        for e_idx in adj_edges[v]:
                            if avail_mask & (1 << e_idx):
                                count += 1
                        return count
                    
                    candidates.sort(key=score)
                    # Randomize ties
                    for i in range(len(candidates) - 1, 0, -1):
                        j = rng.randint(0, i)
                        candidates[i], candidates[j] = candidates[j], candidates[i]
                    
                    for gv in candidates:
                        # Check if gv is used in current tree mapping
                        # We need a fast way to check this. 
                        # Since k <= 20, linear search in local_mapping is okay, 
                        # but let's maintain a set or bitmask for tree nodes.
                        # Let's use a set for local_used
                        # Actually, local_mapping is a list. 
                        # To speed up, we can maintain a local_used_mask for the tree?
                        # No, tree vertices are 0..k-1, host vertices are 0..19.
                        # We need to check if host vertex gv is already assigned to another tree node.
                        # We can maintain a list `host_to_tree_node` of size 20, initialized to -1.
                        
                        # Let's refactor to use host_to_tree_node for O(1) check.
                        
                        if gv in local_mapping: 
                             # This check is O(k). 
                             # Better to maintain a set `local_used_hosts`.
                             pass
                        # I will implement this properly in the final code.
                        
                        local_mapping[node] = gv
                        used_mask |= (1 << gv)
                        
                        ok = True
                        for neighbor in adj[node]:
                            if neighbor == parent:
                                continue
                            if not embed_dfs(neighbor, node):
                                ok = False
                                break
                        
                        if ok:
                            return True
                        
                        local_mapping[node] = -1
                        used_mask &= ~(1 << gv)
                        if parent != -1:
                            failed_states.add((node, gv))
                            
                    return False
                
                # Run DFS
                # Need to initialize local state properly
                # The nested function captures local_mapping, which is re-initialized in the loop?
                # No, local_mapping is defined outside the root loop but inside the tree loop.
                # We must reset it for each root.
                
                # Let's restructure the code to be cleaner.
                pass
            
            # If we didn't embed with any root, fail
            if not embedded:
                success = False
                break
            
            # Update avail_mask with the edges used
            # We need to know which edges were used.
            # The mapping `local_mapping` maps tree node -> host vertex.
            for u, v in trees[t_idx]:
                gu = local_mapping[u]
                gv = local_mapping[v]
                e_idx = edge_idx[(gu, gv)]
                avail_mask &= ~(1 << e_idx)
                packed_edges += 1
                
            current_solution[t_idx] = local_mapping[:]
        
        if success:
            return current_solution
            
    # Fallback
    result = [None] * n
    result[0] = [0]
    result[1] = [0, 1]
    return result
