import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    n = 20
    
    # Pre-process trees
    tree_infos = []
    for i, edges in enumerate(trees):
        k = i + 1
        if k == 1:
            tree_infos.append((edges, {}, [0], 1, 0))
            continue
        adj = {v: [] for v in range(k)}
        degs = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degs[u] += 1
            degs[v] += 1
        max_deg = max(degs)
        tree_infos.append((edges, adj, degs, k, max_deg))
    
    # Sort tree indices: largest max degree first, then largest size
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: (-tree_infos[i][4], -len(trees[i])))
    
    # Build edge index mapping for fast lookup
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    total_edges = 190
    
    # Precompute edges needed per tree
    tree_edge_counts = [len(trees[i]) for i in range(20)]
    # Remaining edges needed from position p onward
    rem_edges_needed = [0] * 21
    for p in range(19, -1, -1):
        rem_edges_needed[p] = rem_edges_needed[p + 1] + tree_edge_counts[tree_indices[p]]
    
    # State: available edges as a set of indices
    all_edge_indices = set(range(total_edges))
    
    best_result = [None] * 20
    best_score = 0
    
    def update_best(mappings):
        nonlocal best_score
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += len(trees[i])
        if score > best_score:
            best_score = score
            best_result = [list(m) if m is not None else None for m in mappings]
    
    def backtrack(tree_pos, available_edges, avail_deg, mappings):
        if time.time() > deadline:
            return False
        
        if tree_pos == 20:
            update_best(mappings)
            return True
        
        t_idx = tree_indices[tree_pos]
        edges, adj, degs, k, max_deg = tree_infos[t_idx]
        
        if k == 1:
            mappings[t_idx] = [0]
            if backtrack(tree_pos + 1, available_edges, avail_deg, mappings):
                return True
            mappings[t_idx] = None
            return False
        
        # Pruning: Check if max available degree >= max tree degree
        if max(avail_deg) < max_deg:
            return False
        
        # Pruning: Check if enough edges remain
        if len(available_edges) < rem_edges_needed[tree_pos]:
            return False
            
        # Vertex ordering: sort by degree descending for better pruning
        tv_order = sorted(range(k), key=lambda v: -degs[v])
        host_map = {}
        
        def assign_tv(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # All tree vertices assigned. Verify and remove edges
                new_avail = set(available_edges)
                new_deg = list(avail_deg)
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    e = (min(hu, hv), max(hu, hv))
                    eidx = edge_to_idx.get(e)
                    if eidx is None or eidx not in new_avail:
                        return False
                    new_avail.discard(eidx)
                    new_deg[hu] -= 1
                    new_deg[hv] -= 1
                
                temp_map = [0] * k
                for tv, hv in host_map.items():
                    temp_map[tv] = hv
                mappings[t_idx] = temp_map
                
                if backtrack(tree_pos + 1, new_avail, new_deg, mappings):
                    return True
                
                mappings[t_idx] = None
                return False
            
            tv = tv_order[pos]
            deg_t = degs[tv]
            
            used_hosts = set(host_map.values())
            
            # Candidates sorted by available degree descending
            candidates = [h for h in range(n) if h not in used_hosts and avail_deg[h] >= deg_t]
            candidates.sort(key=lambda h: -avail_deg[h])
            
            for h in candidates:
                # Check neighbors: for each already-assigned neighbor, edge must be available
                valid = True
                for nb in adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        e = (min(h, hn), max(h, hn))
                        eidx = edge_to_idx.get(e)
                        if eidx is None or eidx not in available_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                host_map[tv] = h
                if assign_tv(pos + 1):
                    return True
                del host_map[tv]
            
            return False
        
        # Symmetry breaking for first tree
        if tree_pos == 0:
            tv0 = tv_order[0]
            host_map[tv0] = 0
            if assign_tv(1):
                return True
            del host_map[tv0]
        else:
            if assign_tv(0):
                return True
        
        return False
    
    # Initial run
    mappings = [None] * 20
    avail_deg_init = [0] * n
    for eidx in all_edge_indices:
        u, v = idx_to_edge[eidx]
        avail_deg_init[u] += 1
        avail_deg_init[v] += 1
    
    if backtrack(0, all_edge_indices, avail_deg_init, mappings):
        result = [None] * 20
        for i in range(20):
            if mappings[i] is not None:
                result[i] = [int(x) for x in mappings[i]]
        return result
    else:
        # Try randomized restarts if time permits
        rng = Random(42)
        while time.time() < deadline - 2:
            mappings = [None] * 20
            # Shuffle candidate ordering by adding randomness
            # We'll do a simple restart with the same deterministic search
            # but we can vary the symmetry-breaking choice
            avail_deg_init = [0] * n
            for eidx in all_edge_indices:
                u, v = idx_to_edge[eidx]
                avail_deg_init[u] += 1
                avail_deg_init[v] += 1
            
            if backtrack(0, all_edge_indices, avail_deg_init, mappings):
                result = [None] * 20
                for i in range(20):
                    if mappings[i] is not None:
                        result[i] = [int(x) for x in mappings[i]]
                return result
        
        # Return best partial result
        return best_result
