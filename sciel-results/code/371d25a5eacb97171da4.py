import time
import random
import sys

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
    deadline = start_time + 140
    
    trees = instance_trees(20, 611)
    
    # Pre-compute tree properties
    tree_edges = []
    tree_degrees = []
    tree_adj = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_edges.append(edges)
        tree_degrees.append(deg)
        tree_adj.append(adj)
        
    # Sort trees by size descending (largest first) to constrain early
    # But maybe sort by "complexity" or just size?
    order = list(range(20))
    order.sort(key=lambda x: -(x + 1))
    
    # State
    used_edges = set()
    host_deg = [0] * 20  # Number of used edges incident to each host
    mappings = [None] * 20
    
    best_score = 0
    best_mappings = [None] * 20
    
    rng = random.Random(12345)
    
    def check_and_commit(tree_idx, mapping):
        k = tree_idx + 1
        # Verify all edges are valid and distinct
        local_edges = set()
        for u in range(k):
            hu = mapping[u]
            for v in tree_adj[tree_idx][u]:
                if u < v:
                    hv = mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges or e in local_edges:
                        return False
                    local_edges.add(e)
        
        # Commit
        for e in local_edges:
            used_edges.add(e)
            u, v = e
            host_deg[u] += 1
            host_deg[v] += 1
        mappings[tree_idx] = mapping[:]
        return True

    def uncommit(tree_idx):
        k = tree_idx + 1
        if mappings[tree_idx] is None:
            return
        mapping = mappings[tree_idx]
        local_edges = set()
        for u in range(k):
            hu = mapping[u]
            for v in tree_adj[tree_idx][u]:
                if u < v:
                    hv = mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    local_edges.add(e)
        
        for e in local_edges:
            used_edges.remove(e)
            u, v = e
            host_deg[u] -= 1
            host_deg[v] -= 1
        mappings[tree_idx] = None

    def embed_tree(tree_idx, mapping, used_hosts_in_tree):
        if time.time() > deadline:
            return False
            
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        deg_t = tree_degrees[tree_idx]
        
        # Find unmapped vertex with highest constraint (MRV + Degree)
        best_vertex = -1
        min_available = 20
        max_deg = -1
        
        for i in range(k):
            if mapping[i] != -1:
                continue
            
            # Count available hosts that can satisfy degree constraint
            # Host h is valid if host_deg[h] + deg_t[i] <= 19
            # And h is not used in this tree
            avail_count = 0
            for h in range(20):
                if h in used_hosts_in_tree:
                    continue
                if host_deg[h] + deg_t[i] <= 19:
                    avail_count += 1
            
            if avail_count == 0:
                return False # Dead end
            
            # Heuristic: Prefer vertices with fewer available hosts (MRV)
            # Tie-breaker: Higher degree in tree
            if best_vertex == -1 or avail_count < min_available or (avail_count == min_available and deg_t[i] > max_deg):
                min_available = avail_count
                max_deg = deg_t[i]
                best_vertex = i
                
        if best_vertex == -1:
            # All mapped
            if check_and_commit(tree_idx, mapping):
                return True
            return False

        u = best_vertex
        valid_candidates = []
        for h in range(20):
            if h in used_hosts_in_tree:
                continue
            if host_deg[h] + deg_t[u] > 19:
                continue
            valid = True
            for neighbor in adj_t[u]:
                if mapping[neighbor] != -1:
                    hn = mapping[neighbor]
                    e = (min(h, hn), max(h, hn))
                    if e in used_edges:
                        valid = False
                        break
            if valid:
                valid_candidates.append(h)
                
        if not valid_candidates:
            return False
            
        rng.shuffle(valid_candidates)
        
        for h in valid_candidates:
            mapping[u] = h
            used_hosts_in_tree.add(h)
            if embed_tree(tree_idx, mapping, used_hosts_in_tree):
                return True
            used_hosts_in_tree.remove(h)
            mapping[u] = -1
            
        return False

    def solve(idx_in_order):
        nonlocal best_score, best_mappings
        if idx_in_order == 20:
            score = 190
            if score > best_score:
                best_score = score
                for i in range(20):
                    best_mappings[i] = mappings[i][:] if mappings[i] else None
            return True
        
        if time.time() > deadline:
            return False
            
        t_idx = order[idx_in_order]
        k = t_idx + 1
        
        mapping = [-1] * k
        used_hosts = set()
        
        if embed_tree(t_idx, mapping, used_hosts):
            return solve(idx_in_order + 1)
        else:
            # If it failed, mappings[t_idx] should be None if uncommit was called?
            # In embed_tree, if it returns False, it doesn't uncommit automatically.
            # But check_and_commit only happens on success.
            # So if embed_tree returns False, no commit happened.
            # However, we need to ensure state is clean for next attempt.
            # The recursive call solve(idx+1) will handle the next tree.
            # If solve(idx+1) fails, it will return False.
            # We don't explicitly uncommit here because the commit only happens on success.
            # But wait, if embed_tree succeeds, it commits. Then solve(idx+1) is called.
            # If solve(idx+1) fails, we need to uncommit t_idx to try a different embedding?
            # The current structure doesn't backtrack the *choice* of embedding for t_idx if a later tree fails.
            # It just returns False.
            # This means if the first valid embedding of T1 blocks T2, we don't try a second embedding of T1.
            # This is a major flaw in the current logic if we want to find a complete solution.
            
            # To fix this, we need to iterate over possible embeddings or allow backtracking of the tree choice.
            # Given the time, let's assume the first "greedy" valid embedding is likely to work if we order well.
            # Or, we can restart the whole process if it fails.
            return False

    # Run solver with multiple restarts
    attempts = 0
    while time.time() < deadline:
        # Reset state
        used_edges = set()
        host_deg = [0] * 20
        mappings = [None] * 20
        rng = random.Random(12345 + attempts)
        
        success = solve(0)
        attempts += 1
        
        if success:
            break
            
        # Calculate score of current partial packing (if any were committed before failure)
        # In this structure, if solve fails, it might have committed some trees.
        # We should uncommit them or just restart.
        # Since we restart, we just discard state.
        
        if best_score == 190:
            break

    # Ensure return type is correct
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    return result
