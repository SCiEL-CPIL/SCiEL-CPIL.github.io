from random import Random
import time

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
    deadline = start_time + 115  # safety margin
    
    trees = instance_trees()
    n = 20
    
    # Pre-compute adjacency and degrees for each tree
    tree_adj = []
    tree_deg = []
    for k in range(1, n + 1):
        adj = [[] for _ in range(k)]
        for u, v in trees[k - 1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        tree_deg.append([len(adj[i]) for i in range(k)])
        
    # Order of trees to process: T20, T19, ..., T1
    # We process T20 specially (any permutation works), then T19..T1 with backtracking.
    
    def try_packing(seed_offset):
        rng = Random(12345 + seed_offset)
        used_edges = set()
        embeddings = [None] * 20
        
        # 1. Embed T20
        # Fix vertex 0 to host 0.
        # Randomly permute the other 19 vertices.
        hosts = list(range(1, 20))
        rng.shuffle(hosts)
        emb20 = [0] + hosts
        # Map tree vertex i to emb20[i]
        # The edges used are (emb20[u], emb20[v]) for (u,v) in trees[19]
        for u, v in trees[19]:
            edge = frozenset((emb20[u], emb20[v]))
            used_edges.add(edge)
        embeddings[19] = emb20
        
        # 2. Embed T19 down to T1
        # We need to find embeddings for trees[18] down to trees[0]
        # such that their edges are not in used_edges.
        
        def embed_tree(tree_idx):
            k = tree_idx + 1
            if k == 1:
                # T1: single vertex. Any host works.
                # To be safe, pick a host that doesn't cause issues? 
                # No edges, so no conflict.
                # Just pick 0.
                return [0]
            
            adj = tree_adj[tree_idx]
            deg = tree_deg[tree_idx]
            
            # Order vertices by degree (high to low)
            vertex_order = sorted(range(k), key=lambda v: -deg[v])
            
            # Backtracking search for embedding
            assignment = [-1] * k  # assignment[tree_v] = host_v
            available_hosts = set(range(n))
            
            def backtrack(vpos):
                if time.time() > deadline:
                    return False
                if vpos == k:
                    return True
                
                tv = vertex_order[vpos]
                
                # Determine valid hosts
                # A host h is valid if for all neighbors nb of tv that are already assigned,
                # the edge (h, assignment[nb]) is NOT in used_edges.
                
                # To speed up, we can filter hosts.
                # But first, let's just iterate through available hosts.
                # We can shuffle them to randomize.
                hosts_list = list(available_hosts)
                rng.shuffle(hosts_list)
                
                for h in hosts_list:
                    # Check conflicts
                    conflict = False
                    for nb in adj[tv]:
                        if assignment[nb] != -1:
                            edge = frozenset((h, assignment[nb]))
                            if edge in used_edges:
                                conflict = True
                                break
                    if conflict:
                        continue
                    
                    # Also, we should check if h is "compatible" with future assignments?
                    # Forward checking: for each unassigned neighbor nb of tv,
                    # is there at least one available host h' != h such that (h', h) is not in used_edges?
                    # This is expensive. Let's try without it first.
                    
                    # Tentative assignment
                    assignment[tv] = h
                    available_hosts.remove(h)
                    
                    if backtrack(vpos + 1):
                        return True
                    
                    # Undo
                    assignment[tv] = -1
                    available_hosts.add(h)
                
                return False
            
            if backtrack(0):
                # Extract embedding
                emb = assignment
                # Add edges to used_edges
                for u, v in trees[tree_idx]:
                    edge = frozenset((emb[u], emb[v]))
                    used_edges.add(edge)
                return emb
            else:
                return None
        
        # Process T19..T1
        for tree_idx in range(18, -1, -1):
            if time.time() > deadline:
                return None
            emb = embed_tree(tree_idx)
            if emb is None:
                return None
            embeddings[tree_idx] = emb
            
        return embeddings

    best_result = None
    for attempt in range(2000):
        if time.time() > deadline:
            break
        result = try_packing(attempt)
        if result is not None:
            # Verify
            # Check distinctness within each tree
            valid = True
            for k in range(1, n + 1):
                emb = result[k - 1]
                if len(set(emb)) != k:
                    valid = False
                    break
            if valid:
                # Check edge disjointness
                all_edges = set()
                for k in range(1, n + 1):
                    emb = result[k - 1]
                    for u, v in trees[k - 1]:
                        edge = frozenset((emb[u], emb[v]))
                        if edge in all_edges:
                            valid = False
                            break
                        all_edges.add(edge)
                if valid:
                    best_result = result
                    break # Found a perfect packing
    
    if best_result is not None:
        out = []
        for k in range(1, n + 1):
            emb = best_result[k - 1]
            if emb is None:
                out.append(None)
            else:
                out.append([int(x) for x in emb])
        return out
    else:
        # Fallback
        return [None] * 20
