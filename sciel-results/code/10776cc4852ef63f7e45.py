import time
import random

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
    deadline = start_time + 130  # Safety margin before 150s kill
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_hosts = 20
    
    # Pre-compute tree structure
    tree_adj = []
    tree_degrees = []
    tree_edges_list = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_degrees.append(deg)
        tree_edges_list.append(edges)
    
    # Total edges to pack
    total_edges_needed = sum(len(t) for t in tree_edges_list)
    
    # All possible edges in K20
    all_k20_edges = set()
    for i in range(n_hosts):
        for j in range(i + 1, n_hosts):
            all_k20_edges.add((i, j))
    
    # State for current attempt
    used_edges = set()
    mappings = [None] * n_trees
    host_used_degree = [0] * n_hosts  # Degree used at each host vertex
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def get_tree_edges_as_host_edges(tree_idx, mapping):
        """Get the set of host edges used by a tree given its mapping."""
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        edges = set()
        for u in range(k):
            hu = mapping[u]
            for v in adj_t[u]:
                if u < v:
                    hv = mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    edges.add(e)
        return edges
    
    def check_tree_valid(tree_idx, mapping):
        """Check if a tree mapping is valid (edge-disjoint, degree constraints)."""
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Check distinctness
        seen = set()
        for i in range(k):
            if mapping[i] in seen:
                return False
            seen.add(mapping[i])
        
        # Check edges
        for u in range(k):
            hu = mapping[u]
            for v in adj_t[u]:
                if u < v:
                    hv = mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        return False
        
        # Check degree constraints
        host_deg_in_tree = [0] * n_hosts
        for u in range(k):
            hu = mapping[u]
            host_deg_in_tree[hu] += tree_degrees[tree_idx][u]
        
        for h in range(n_hosts):
            if host_used_degree[h] + host_deg_in_tree[h] > 19:
                return False
        
        return True
    
    def embed_tree_csp(tree_idx):
        """Embed a tree using backtracking with MRV (Minimum Remaining Values) heuristic."""
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        deg_t = tree_degrees[tree_idx]
        
        # Start with a root vertex (vertex 0)
        mapping = [-1] * k
        used_in_tree = set()
        
        # Try different roots
        for root_host in range(n_hosts):
            if host_used_degree[root_host] + deg_t[0] > 19:
                continue
            mapping[0] = root_host
            used_in_tree.add(root_host)
            
            # Backtracking search
            def backtrack():
                if time.time() > deadline:
                    return False
                
                # Find next unmapped vertex with fewest valid host options (MRV)
                best_vertex = -1
                best_options = None
                best_count = float('inf')
                
                for i in range(k):
                    if mapping[i] != -1:
                        continue
                    
                    # Count valid hosts for this vertex
                    valid_hosts = []
                    for h in range(n_hosts):
                        if h in used_in_tree:
                            continue
                        if host_used_degree[h] + deg_t[i] > 19:
                            continue
                        
                        # Check if edges to already-mapped neighbors would conflict
                        valid = True
                        for neighbor in adj_t[i]:
                            if mapping[neighbor] != -1:
                                hn = mapping[neighbor]
                                e = (min(h, hn), max(h, hn))
                                if e in used_edges:
                                    valid = False
                                    break
                        
                        if valid:
                            valid_hosts.append(h)
                    
                    if len(valid_hosts) < best_count:
                        best_count = len(valid_hosts)
                        best_vertex = i
                        best_options = valid_hosts
                        
                        if best_count == 0:
                            break
                        if best_count == 1:
                            break
                
                if best_vertex == -1:
                    # All vertices mapped
                    # Final validation
                    if check_tree_valid(tree_idx, mapping):
                        # Commit
                        for u in range(k):
                            hu = mapping[u]
                            for v in adj_t[u]:
                                if u < v:
                                    hv = mapping[v]
                                    e = (min(hu, hv), max(hu, hv))
                                    used_edges.add(e)
                            host_used_degree[hu] += deg_t[u]
                        mappings[tree_idx] = mapping[:]
                        return True
                    return False
                
                # Try each valid host for best_vertex
                v = best_vertex
                # Shuffle for randomness
                random.shuffle(best_options)
                
                for h in best_options:
                    mapping[v] = h
                    used_in_tree.add(h)
                    
                    if backtrack():
                        return True
                    
                    used_in_tree.remove(h)
                    mapping[v] = -1
                
                return False
            
            if backtrack():
                return True
        
        return False
    
    # Sort trees by size (largest first for better pruning)
    order = list(range(n_trees))
    order.sort(key=lambda x: -(x + 1))
    
    # Multiple restarts with different random seeds
    attempts = 0
    while time.time() < deadline:
        # Reset state
        used_edges = set()
        mappings = [None] * n_trees
        host_used_degree = [0] * n_hosts
        random.seed(12345 + attempts)
        
        success = True
        for idx in order:
            if time.time() > deadline:
                break
            if not embed_tree_csp(idx):
                success = False
                break
        
        # Calculate score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges_list[i])
        
        if score > best_score:
            best_score = score
            for i in range(n_trees):
                best_mappings[i] = mappings[i][:] if mappings[i] else None
        
        attempts += 1
        
        if best_score == total_edges_needed:
            break
        
        if not success:
            # Try a different order on next attempt
            if attempts % 3 == 0:
                order = list(range(n_trees))
                random.shuffle(order)
    
    # Ensure return type is correct
    result = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    return result
