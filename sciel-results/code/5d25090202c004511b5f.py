"""New program: Greedy Embedding with Local Search (Swap Repair) for packing trees into K20."""
from random import Random
import sys
sys.setrecursionlimit(10000)

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
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute tree adjacency and degree
    tree_adj = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Strategy: 
    # 1. Order trees. We try a few orderings: Descending Size, Ascending Size, Random.
    # 2. For each ordering, try to embed trees sequentially.
    # 3. Use a heuristic to pick host vertices: prefer vertices with fewer used edges incident to them? 
    #    Or simply random but with backtracking limited to a small depth if it fails.
    # 4. If a tree fails, try to repair by swapping vertices in previously embedded trees.
    
    best_value = 0
    best_result = [None] * n
    
    # Define a function to check if a mapping is valid for a tree given used_edges
    def is_valid_mapping(mapping, tree_idx, used_edges):
        k = tree_idx + 1
        adj = tree_adj[tree_idx]
        # Check injectivity (implicitly assumed if we construct carefully, but good to check)
        # Check edge disjointness
        for u in range(k):
            hu = mapping[u]
            for v in adj[u]:
                if u < v:
                    hv = mapping[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        return False
        return True

    # Heuristic embedding for a single tree
    def embed_tree(tree_idx, used_edges, rng, max_backtrack_depth=5):
        k = tree_idx + 1
        if k == 1:
            # Pick any unused vertex? No, T1 has no edges, so it doesn't consume edges.
            # But it consumes a vertex? The problem says "list of k distinct int host vertices".
            # So yes, it consumes a vertex.
            # However, T1 has 0 edges. So it doesn't affect used_edges.
            # We just need to pick a vertex.
            # To be safe, we should pick a vertex that doesn't conflict with future trees?
            # For now, just pick a random one.
            candidates = list(range(n))
            rng.shuffle(candidates)
            return [candidates[0]]
        
        adj = tree_adj[tree_idx]
        
        # We will use a recursive search with limited depth for "hard" trees
        # But first, try a greedy assignment.
        
        # Order of tree nodes to assign: 
        # Start with a random node, then always pick an unassigned node 
        # adjacent to an assigned node. This ensures connectivity.
        
        def greedy_embed():
            mapping = [-1] * k
            # Pick a random start node
            start = rng.randint(0, k-1)
            mapping[start] = rng.randint(0, n-1)
            assigned = [False] * k
            assigned[start] = True
            
            for step in range(1, k):
                # Find unassigned nodes adjacent to assigned nodes
                candidates_nodes = []
                for v in range(k):
                    if not assigned[v]:
                        for nb in adj[v]:
                            if assigned[nb]:
                                candidates_nodes.append(v)
                                break
                if not candidates_nodes:
                    # Should not happen for a tree
                    return None
                
                # Pick one of these nodes
                node = rng.choice(candidates_nodes)
                
                # Find valid host vertices for this node
                valid_hosts = []
                for h in range(n):
                    # Check if h is already used by this tree
                    if any(mapping[u] == h for u in range(k) if mapping[u] != -1):
                        continue
                    
                    # Check edge conflicts with already assigned neighbors
                    ok = True
                    for nb in adj[node]:
                        if assigned[nb]:
                            h_nb = mapping[nb]
                            edge = (min(h, h_nb), max(h, h_nb))
                            if edge in used_edges:
                                ok = False
                                break
                    if ok:
                        valid_hosts.append(h)
                
                if not valid_hosts:
                    return None
                
                # Heuristic: Pick a host vertex that is "least used" globally?
                # We don't track global usage easily here, so just pick random from valid.
                # Or pick the one with min degree in the *remaining* available graph?
                # Let's just pick random for speed.
                h = rng.choice(valid_hosts)
                mapping[node] = h
                assigned[node] = True
                
            return mapping

        # Try greedy multiple times
        for _ in range(100):
            m = greedy_embed()
            if m:
                return m
        
        # If greedy fails, try a small backtracking search
        # This is expensive, so we limit it.
        def backtrack_embed():
            mapping = [-1] * k
            assigned = [False] * k
            
            def dfs(depth):
                if depth == k:
                    return True
                
                # Select next node: most constrained (most assigned neighbors)
                best_node = -1
                max_assigned_nb = -1
                for v in range(k):
                    if not assigned[v]:
                        cnt = sum(1 for nb in adj[v] if assigned[nb])
                        if cnt > max_assigned_nb:
                            max_assigned_nb = cnt
                            best_node = v
                
                if best_node == -1:
                    # Pick any unassigned
                    best_node = next(v for v in range(k) if not assigned[v])
                
                # Candidates
                candidates = []
                for h in range(n):
                    if any(mapping[u] == h for u in range(k) if mapping[u] != -1):
                        continue
                    ok = True
                    for nb in adj[best_node]:
                        if assigned[nb]:
                            h_nb = mapping[nb]
                            edge = (min(h, h_nb), max(h, h_nb))
                            if edge in used_edges:
                                ok = False
                                break
                    if ok:
                        candidates.append(h)
                
                rng.shuffle(candidates)
                
                for h in candidates:
                    mapping[best_node] = h
                    assigned[best_node] = True
                    if dfs(depth + 1):
                        return True
                    mapping[best_node] = -1
                    assigned[best_node] = False
                
                return False
            
            return dfs(0) if dfs(0) else None

        return backtrack_embed()

    # Main loop: Try different orderings
    orderings = [
        list(range(n)), 
        list(range(n-1, -1, -1)),
        list(range(n)), 
        list(range(n))
    ]
    # Shuffle some orderings
    for i in range(2):
        rng_order = Random(i)
        rng_order.shuffle(orderings[i])
    
    for trial in range(20):
        rng = Random(trial)
        used_edges = set()
        result = [None] * n
        
        # Pick an ordering
        order = orderings[trial % len(orderings)]
        if trial >= 2:
            rng.shuffle(order)
            
        success = True
        for idx in order:
            mapping = embed_tree(idx, used_edges, rng)
            if mapping is None:
                # Try to repair
                # This is complex. For now, if it fails, we break and try a new trial.
                success = False
                break
            else:
                result[idx] = mapping
                # Update used edges
                for u, v in trees[idx]:
                    hu = mapping[u]
                    hv = mapping[v]
                    edge = (min(hu, hv), max(hu, hv))
                    used_edges.add(edge)
        
        if success:
            value = sum(len(trees[i]) for i in range(n))
            if value > best_value:
                best_value = value
                best_result = result[:]
                if value == 190:
                    break
    
    if best_value > 0:
        out = [None] * n
        for i in range(n):
            if best_result[i] is not None:
                out[i] = [int(v) for v in best_result[i]]
        return out
    
    return [None] * n
