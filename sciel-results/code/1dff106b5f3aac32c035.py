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
    deadline = start_time + 115
    
    trees = instance_trees()
    N_HOSTS = 20
    
    # Precompute tree structures: for each tree, get adjacency and degree
    tree_info = []
    for i in range(20):
        k = i + 1
        if k <= 2:
            tree_info.append({'k': k, 'edges': trees[i], 'adj': {}, 'leaves': []})
            continue
        adj = {v: [] for v in range(k)}
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        tree_info.append({'k': k, 'edges': trees[i], 'adj': adj, 'leaves': leaves})
    
    def try_backtrack(order=None):
        """Try to pack all trees using backtracking with leaf-first ordering."""
        if time.time() > deadline:
            return None
        
        # Order: pack larger trees first (they're more constrained)
        if order is None:
            order = list(range(19, -1, -1))  # T20, T19, ..., T1
        
        # State: used_edges set, and mapping for each tree
        used_edges = set()
        mappings = [None] * 20
        
        # For backtracking, we need to assign host vertices to tree vertices
        # We'll do this recursively, placing one tree at a time
        
        def place_tree(tree_idx, assigned, remaining_hosts):
            """Place tree tree_idx given already-assigned vertices and remaining hosts."""
            if time.time() > deadline:
                return None
            
            info = tree_info[tree_idx]
            k = info['k']
            
            if k == 1:
                # Single vertex: pick any available host
                if remaining_hosts:
                    host = remaining_hosts[0]
                    new_remaining = remaining_hosts[1:]
                    mappings[tree_idx] = [host]
                    result = place_next(tree_idx + 1, new_remaining)
                    if result is not None:
                        return result
                    mappings[tree_idx] = None
                return None
            
            # For k >= 2, we need to assign k distinct hosts to k tree vertices
            # Use a recursive approach: pick a root, then BFS/DFS to assign neighbors
            
            # Try different roots (prefer high-degree vertices for constraint propagation)
            adj = info['adj']
            # Sort vertices by degree descending for better constraint propagation
            vertices_by_deg = sorted(range(k), key=lambda v: -len(adj[v]))
            
            for root in vertices_by_deg:
                # Try to build a valid embedding with this root
                # We'll use a recursive assignment that respects edge constraints
                
                def assign_embedding(root, partial_map, remaining):
                    """Recursively assign hosts to tree vertices starting from root."""
                    if time.time() > deadline:
                        return None
                    
                    if len(partial_map) == k:
                        # Check if all edges are valid (not in used_edges)
                        for u, v in info['edges']:
                            e1 = min(partial_map[u], partial_map[v])
                            e2 = max(partial_map[u], partial_map[v])
                            if (e1, e2) in used_edges:
                                return None
                        # Valid embedding found
                        mappings[tree_idx] = list(partial_map.values())
                        # Wait, we need to return the mapping properly
                        # partial_map is {tree_vertex: host}
                        mapping_list = [0] * k
                        for tv, h in partial_map.items():
                            mapping_list[tv] = h
                        
                        # Save and recurse
                        old_mapping = mappings[tree_idx]
                        mappings[tree_idx] = mapping_list
                        # Mark edges as used
                        new_used = set(used_edges)
                        for u, v in info['edges']:
                            e1 = min(mapping_list[u], mapping_list[v])
                            e2 = max(mapping_list[u], mapping_list[v])
                            new_used.add((e1, e2))
                        
                        result = place_next(tree_idx + 1, new_used, remaining)
                        if result is not None:
                            return result
                        
                        # Backtrack
                        mappings[tree_idx] = old_mapping
                        return None
                    
                    # Find next unassigned vertex
                    unassigned = [v for v in range(k) if v not in partial_map]
                    if not unassigned:
                        return None
                    
                    # Choose vertex with most constraints (most assigned neighbors)
                    best_v = None
                    best_score = -1
                    for v in unassigned:
                        assigned_neighbors = [n for n in adj[v] if n in partial_map]
                        score = len(assigned_neighbors)
                        if score > best_score:
                            best_score = score
                            best_v = v
                    
                    if best_v is None:
                        return None
                    
                    # Try each available host for best_v
                    for host in remaining:
                        # Check if this assignment is consistent with already-assigned neighbors
                        valid = True
                        for n in adj[best_v]:
                            if n in partial_map:
                                e1 = min(host, partial_map[n])
                                e2 = max(host, partial_map[n])
                                if (e1, e2) in used_edges:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        new_partial = dict(partial_map)
                        new_partial[best_v] = host
                        new_remaining = [h for h in remaining if h != host]
                        
                        result = assign_embedding(root, new_partial, new_remaining)
                        if result is not None:
                            return result
                    
                    return None
                
                result = assign_embedding(root, {root: None}, list(remaining_hosts))
                # Wait, I need to fix this - the initial call should assign a host to root
                # Let me restructure
                
            return None
        
        # Let me restructure the backtracking more cleanly
        def solve():
            """Solve the packing problem using backtracking."""
            if time.time() > deadline:
                return None
            
            used_edges = set()
            mappings = [None] * 20
            remaining_hosts = set(range(N_HOSTS))
            
            def place_tree_recursive(idx, used_edges, remaining_hosts, mappings):
                if time.time() > deadline:
                    return None
                
                if idx == 20:
                    return mappings[:]
                
                info = tree_info[idx]
                k = info['k']
                
                if k == 1:
                    for host in remaining_hosts:
                        new_remaining = remaining_hosts - {host}
                        mappings[idx] = [host]
                        result = place_tree_recursive(idx + 1, used_edges, new_remaining, mappings)
                        if result is not None:
                            return result
                        mappings[idx] = None
                    return None
                
                # For k >= 2: try all possible embeddings
                # This is expensive, so use constraint propagation
                adj = info['adj']
                
                # Build a list of candidate embeddings using a smarter search
                # We'll do a DFS on the tree structure, assigning hosts as we go
                
                def embed_tree(used_edges, remaining_hosts, adj, k, edges):
                    """Find a valid embedding of this tree."""
                    if time.time() > deadline:
                        return None
                    
                    # Choose root: highest degree vertex
                    root = max(range(k), key=lambda v: len(adj[v]))
                    
                    # DFS to assign hosts
                    def dfs_assign(partial_map, remaining, current_vertex, parent):
                        if time.time() > deadline:
                            return None
                        
                        if len(partial_map) == k:
                            # Verify all edges
                            for u, v in edges:
                                e1 = min(partial_map[u], partial_map[v])
                                e2 = max(partial_map[u], partial_map[v])
                                if (e1, e2) in used_edges:
                                    return None
                            mapping = [0] * k
                            for tv, h in partial_map.items():
                                mapping[tv] = h
                            return mapping
                        
                        # Find next vertex to assign
                        # Prefer vertices with assigned neighbors for constraint propagation
                        unassigned = [v for v in range(k) if v not in partial_map]
                        if not unassigned:
                            return None
                        
                        # Score by number of assigned neighbors
                        scored = []
                        for v in unassigned:
                            assigned_neigh = [n for n in adj[v] if n in partial_map]
                            scored.append((len(assigned_neigh), v))
                        scored.sort(reverse=True)
                        
                        next_v = scored[0][1]
                        
                        for host in sorted(remaining, key=lambda h: h):
                            # Check edge constraints with assigned neighbors
                            valid = True
                            for n in adj[next_v]:
                                if n in partial_map:
                                    e1 = min(host, partial_map[n])
                                    e2 = max(host, partial_map[n])
                                    if (e1, e2) in used_edges:
                                        valid = False
                                        break
                            if not valid:
                                continue
                            
                            new_partial = dict(partial_map)
                            new_partial[next_v] = host
                            new_remaining = remaining - {host}
                            
                            result = dfs_assign(new_partial, new_remaining, next_v, next_v)
                            if result is not None:
                                return result
                        
                        return None
                    
                    # Start with root unassigned, try all hosts for root
                    for host in sorted(remaining_hosts, key=lambda h: h):
                        partial = {root: host}
                        remaining_after = remaining_hosts - {host}
                        result = dfs_assign(partial, remaining_after, root, -1)
                        if result is not None:
                            return result
                    
                    return None
                
                embedding = embed_tree(used_edges, remaining_hosts, adj, k, info['edges'])
                if embedding is None:
                    return None
                
                mappings[idx] = embedding
                new_used = set(used_edges)
                for u, v in info['edges']:
                    e1 = min(embedding[u], embedding[v])
                    e2 = max(embedding[u], embedding[v])
                    new_used.add((e1, e2))
                
                new_remaining = remaining_hosts - set(embedding)
                
                result = place_tree_recursive(idx + 1, new_used, new_remaining, mappings)
                if result is not None:
                    return result
                
                return None
            
            return place_tree_recursive(0, used_edges, remaining_hosts, mappings)
        
        # Try packing from largest to smallest
        order = list(range(19, -1, -1))
        
        # Reorder tree_info to match
        reordered_info = [tree_info[i] for i in order]
        reordered_trees = [trees[i] for i in order]
        
        # Use the solver with reordered trees
        used_edges = set()
        mappings_reordered = [None] * 20
        remaining_hosts = set(range(N_HOSTS))
        
        def place_reordered(idx, used_edges, remaining_hosts, mappings_reordered, order, tree_info_list):
            if time.time() > deadline:
                return None
            
            if idx == 20:
                # Convert back to original order
                final_mappings = [None] * 20
                for pos, orig_idx in enumerate(order):
                    final_mappings[orig_idx] = mappings_reordered[pos]
                return final_mappings
            
            orig_idx = order[idx]
            info = tree_info_list[idx]
            k = info['k']
            
            if k == 1:
                for host in sorted(remaining_hosts):
                    new_remaining = remaining_hosts - {host}
                    mappings_reordered[idx] = [host]
                    result = place_reordered(idx + 1, used_edges, new_remaining, mappings_reordered, order, tree_info_list)
                    if result is not None:
                        return result
                    mappings_reordered[idx] = None
                return None
            
            adj = info['adj']
            edges = info['edges']
            
            def embed_tree_local(used_edges, remaining_hosts, adj, k, edges):
                if time.time() > deadline:
                    return None
                
                root = max(range(k), key=lambda v: len(adj[v]))
                
                def dfs_assign(partial_map, remaining):
                    if time.time() > deadline:
                        return None
                    
                    if len(partial_map) == k:
                        for u, v in edges:
                            e1 = min(partial_map[u], partial_map[v])
                            e2 = max(partial_map[u], partial_map[v])
                            if (e1, e2) in used_edges:
                                return None
                        mapping = [0] * k
                        for tv, h in partial_map.items():
                            mapping[tv] = h
                        return mapping
                    
                    unassigned = [v for v in range(k) if v not in partial_map]
                    if not unassigned:
                        return None
                    
                    scored = []
                    for v in unassigned:
                        assigned_neigh = [n for n in adj[v] if n in partial_map]
                        scored.append((len(assigned_neigh), v))
                    scored.sort(reverse=True)
                    
                    next_v = scored[0][1]
                    
                    for host in sorted(remaining):
                        valid = True
                        for n in adj[next_v]:
                            if n in partial_map:
                                e1 = min(host, partial_map[n])
                                e2 = max(host, partial_map[n])
                                if (e1, e2) in used_edges:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        new_partial = dict(partial_map)
                        new_partial[next_v] = host
                        new_remaining = remaining - {host}
                        
                        result = dfs_assign(new_partial, new_remaining)
                        if result is not None:
                            return result
                    
                    return None
                
                for host in sorted(remaining_hosts):
                    partial = {root: host}
                    remaining_after = remaining_hosts - {host}
                    result = dfs_assign(partial, remaining_after)
                    if result is not None:
                        return result
                
                return None
            
            embedding = embed_tree_local(used_edges, remaining_hosts, adj, k, edges)
            if embedding is None:
                return None
            
            mappings_reordered[idx] = embedding
            new_used = set(used_edges)
            for u, v in edges:
                e1 = min(embedding[u], embedding[v])
                e2 = max(embedding[u], embedding[v])
                new_used.add((e1, e2))
            
            new_remaining = remaining_hosts - set(embedding)
            
            result = place_reordered(idx + 1, new_used, new_remaining, mappings_reordered, order, tree_info_list)
            if result is not None:
                return result
            
            return None
        
        result = place_reordered(0, used_edges, remaining_hosts, mappings_reordered, order, reordered_info)
        return result
    
    # First, try deterministic backtracking
    result = try_backtrack()
    if result is not None:
        # Verify it's valid
        used = set()
        valid = True
        total_edges = 0
        for i in range(20):
            k = i + 1
            if k == 1:
                total_edges += 0
                continue
            mapping = result[i]
            if mapping is None or len(mapping) != k or len(set(mapping)) != k:
                valid = False
                break
            for u, v in trees[i]:
                e1 = min(mapping[u], mapping[v])
                e2 = max(mapping[u], mapping[v])
                if (e1, e2) in used:
                    valid = False
                    break
                used.add((e1, e2))
                total_edges += 1
            if not valid:
                break
        
        if valid and total_edges == 190:
            return result
    
    # Fallback: use simulated annealing
    state = [None] * 20
    for i in range(20):
        k = i + 1
        if k == 1:
            state[i] = [random.randint(0, 19)]
        else:
            hosts = list(range(N_HOSTS))
            random.shuffle(hosts)
            state[i] = hosts[:k]
    
    def calc_conflicts(state):
        used_edges = set()
        conflicts = 0
        for i in range(20):
            k = i + 1
            if state[i] is None:
                continue
            mapping = state[i]
            if k == 1:
                continue
            tree_edges = trees[i]
            for u, v in tree_edges:
                e1 = min(mapping[u], mapping[v])
                e2 = max(mapping[u], mapping[v])
                if (e1, e2) in used_edges:
                    conflicts += 1
                else:
                    used_edges.add((e1, e2))
        return conflicts
    
    current_conflicts = calc_conflicts(state)
    best_state = [s[:] if s else None for s in state]
    best_conflicts = current_conflicts
    
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if current_conflicts == 0:
            break
        
        i = random.randint(0, 19)
        k = i + 1
        if k == 1:
            continue
        
        old_mapping = state[i]
        
        strategy = random.random()
        
        if strategy < 0.3:
            new_mapping = old_mapping[:]
            a, b = random.sample(range(k), 2)
            new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif strategy < 0.6:
            new_mapping = old_mapping[:]
            a = random.randint(0, k - 1)
            used_hosts = set(new_mapping)
            available = [h for h in range(N_HOSTS) if h not in used_hosts]
            if available:
                new_mapping[a] = random.choice(available)
            else:
                b = random.randint(0, k - 1)
                new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        else:
            hosts = list(range(N_HOSTS))
            random.shuffle(hosts)
            new_mapping = hosts[:k]
        
        state[i] = new_mapping
        new_conflicts = calc_conflicts(state)
        
        delta = new_conflicts - current_conflicts
        
        if delta < 0 or random.random() < pow(2.718, -delta / temperature):
            current_conflicts = new_conflicts
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_state = [s[:] if s else None for s in state]
        else:
            state[i] = old_mapping
        
        temperature *= cooling
        if temperature < 1e-4:
            temperature = 1e-4
    
    return best_state
