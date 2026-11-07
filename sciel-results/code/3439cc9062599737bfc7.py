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
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    trees = instance_trees(20, 611)
    
    # Pre-process trees
    tree_adj = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Total edges to embed
    total_tree_edges = sum(len(t) for t in trees)
    
    # All edges in K20
    all_k20_edges = set()
    for i in range(20):
        for j in range(i+1, 20):
            all_k20_edges.add((i, j))
    
    # Strategy: Process trees in an order that helps packing.
    # Trees with more edges are more constrained, so do them first.
    # But also consider: trees that are "star-like" are easier to place.
    # Let's try largest first with some randomization.
    
    best_result = [None] * 20
    best_edges = 0
    
    # Pre-compute for each tree: degree sequence and a good vertex ordering
    tree_info = []
    for k in range(1, 21):
        if k <= 1:
            tree_info.append(None)
            continue
        adj = tree_adj[k-1]
        deg = [len(adj[i]) for i in range(k)]
        # Order: high degree first for better pruning
        v_order = sorted(range(k), key=lambda x: (-deg[x], x))
        tree_info.append((adj, v_order))
    
    # Try multiple strategies with different orderings
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(attempt * 7919 + 123456)
        
        # Choose ordering: mostly largest-first, with some variation
        if attempt % 5 == 0:
            order = list(range(20, 0, -1))  # Largest first
        elif attempt % 5 == 1:
            order = list(range(20, 0, -1))
            rng.shuffle(order)
        elif attempt % 5 == 2:
            # Smallest first (might help in some cases)
            order = list(range(1, 21))
        elif attempt % 5 == 3:
            # Medium trees first
            order = list(range(20, 0, -1))
            # Move some small trees to front
            small = [k for k in order if k <= 5]
            rest = [k for k in order if k > 5]
            rng.shuffle(small)
            order = small + rest
        else:
            order = list(range(20, 0, -1))
            rng.shuffle(order)
        
        used_edges = set()
        result = [None] * 20
        success = True
        embedded_count = 0
        embedded_edges = 0
        
        for k in order:
            if time.time() > deadline:
                success = False
                break
                
            if k == 1:
                result[0] = [rng.randint(0, 19)]
                embedded_count += 1
                continue
                
            if k == 2:
                found = False
                # Pick a random unused edge
                available = [e for e in all_k20_edges if e not in used_edges]
                if available:
                    edge = rng.choice(available)
                    used_edges.add(edge)
                    result[1] = [edge[0], edge[1]]
                    found = True
                    embedded_count += 1
                    embedded_edges += 1
                if not found:
                    success = False
                    break
                continue
            
            # For k >= 3
            info = tree_info[k-1]
            if info is None:
                continue
            adj, v_order = info
            k_val = k
            num_edges = k_val - 1
            
            mapping = [-1] * k_val
            host_used = [False] * 20
            
            # Use iterative deepening or just backtracking with good heuristics
            # Key insight: we must use ALL edges of K20, so remaining edges = 190 - len(used_edges)
            # After placing this tree, remaining edges must be >= sum of edges in unplaced trees
            
            remaining_trees = [kk for kk in order if kk not in [k2 for k2 in order[:order.index(k)+1] if result[k2-1] is not None]]
            # Actually simpler: count edges in trees not yet placed
            placed_trees = set()
            for kk in order:
                if kk <= k and result[kk-1] is not None:
                    placed_trees.add(kk)
            # Better: just check if we have enough edges left
            edges_placed_so_far = len(used_edges)
            edges_this_tree = num_edges
            # Edges needed by all other unplaced trees
            other_edges_needed = 0
            for kk in range(1, 21):
                if kk != k and result[kk-1] is None:
                    other_edges_needed += max(0, kk - 1)
            
            # Forward checking: after placing this tree, can we still fit all others?
            # This is a necessary but not sufficient condition
            if edges_placed_so_far + edges_this_tree + other_edges_needed > 190:
                # Not enough edges - but this should never happen if total = 190
                pass
            
            # Backtracking search
            # Use a timeout for this tree
            tree_deadline = time.time() + 5  # 5 seconds per tree max
            node_count = [0]
            max_nodes = 500000  # Limit nodes per tree
            
            def backtrack(idx):
                if time.time() > tree_deadline or node_count[0] > max_nodes:
                    return False
                node_count[0] += 1
                if idx == k_val:
                    return True
                
                tv = v_order[idx]
                
                # Assigned neighbors
                assigned_nbs = [nb for nb in adj[tv] if mapping[nb] != -1]
                
                candidates = []
                for h in range(20):
                    if host_used[h]:
                        continue
                    ok = True
                    for nb in assigned_nbs:
                        hnb = mapping[nb]
                        edge = (min(h, hnb), max(h, hnb))
                        if edge in used_edges:
                            ok = False
                            break
                    if ok:
                        candidates.append(h)
                        
                if not candidates:
                    return False
                
                # Heuristic: prefer candidates that leave more flexibility
                # Count how many unused edges are incident to each candidate
                # Actually, prefer candidates with fewer unused incident edges (more constrained)
                # Or: prefer candidates that are "central" in terms of unused edge connectivity
                
                # Simple heuristic: shuffle for randomization, but bias toward
                # vertices that have many unused edges (more options for future)
                # Actually, let's try: prefer vertices with FEWER unused edges
                # (fail early principle)
                
                def score_candidate(h):
                    # Count unused edges incident to h
                    count = 0
                    for nb in range(20):
                        if nb == h:
                            continue
                        e = (min(h, nb), max(h, nb))
                        if e not in used_edges:
                            count += 1
                    return count
                
                # Sort by score (ascending = fewer unused edges first = more constrained)
                candidates.sort(key=score_candidate)
                
                # Add some randomization
                if rng.random() < 0.3:
                    rng.shuffle(candidates)
                
                for h in candidates:
                    mapping[tv] = h
                    host_used[h] = True
                    
                    if backtrack(idx + 1):
                        return True
                        
                    mapping[tv] = -1
                    host_used[h] = False
                    
                return False
            
            if backtrack(0):
                result[k-1] = [int(mapping[i]) for i in range(k_val)]
                for (u, v) in trees[k-1]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
                embedded_count += 1
                embedded_edges += num_edges
            else:
                success = False
                break
        
        # Record result (even partial)
        if embedded_edges > best_edges:
            best_edges = embedded_edges
            best_result = result[:]
            if best_edges == total_tree_edges:
                break
    
    return best_result
