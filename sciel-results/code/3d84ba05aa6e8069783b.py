import time
import random
import sys

sys.setrecursionlimit(100000)

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
    # 120s limit, use 115s for safety
    deadline = start_time + 115
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20
    
    # Precompute tree adjacencies
    tree_adj = []
    tree_edges = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        tree_adj.append(adj_list)
        tree_edges.append(edges)
        
    # Sort trees by size descending
    # T20 is index 19, T1 is index 0
    # We want to pack largest first
    order = sorted(range(n_trees), key=lambda x: -(x + 1))
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def try_packing(seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(seed)
        
        # used_edges is a set of frozensets or tuples (min, max)
        used_edges = set()
        mappings = [None] * n_trees
        
        # Helper to check if edge (u, v) is available
        def is_edge_available(u, v):
            if u == v: return False
            e = (min(u, v), max(u, v))
            return e not in used_edges
            
        def add_edge(u, v):
            e = (min(u, v), max(u, v))
            used_edges.add(e)
            
        def remove_edge(u, v):
            e = (min(u, v), max(u, v))
            used_edges.remove(e)

        def pack_tree(t_idx):
            k = t_idx + 1
            adj_t = tree_adj[t_idx]
            edges_t = tree_edges[t_idx]
            
            if k == 1:
                # Single vertex, can be anywhere, but must be distinct?
                # "list of k distinct int host vertices"
                # If k=1, just pick any available vertex.
                # Since no edges, any vertex works.
                # But we need distinct vertices for the tree.
                # If we pick a vertex, it's used.
                # For k=1, we just need to assign a host.
                # To be safe, pick the first available vertex.
                for h in range(n_vertices):
                    # No edges to check, so just pick one.
                    # But we need to ensure we don't reuse vertices?
                    # "k distinct int host vertices"
                    # Yes, vertices within the tree must be distinct.
                    # And edge disjointness.
                    # For k=1, no edges.
                    # We can map it to any vertex.
                    # Let's just map to 0 for simplicity, or random.
                    # Actually, if we map to 0, and 0 is used by another tree, is that allowed?
                    # "Embedded trees must be edge-disjoint".
                    # Vertices can be shared.
                    # So for k=1, any vertex is fine.
                    mappings[t_idx] = [0]
                    return True
            
            mapping = [-1] * k
            used_verts = set()
            
            # Order vertices in the tree to map
            # We will use backtracking with MRV (Minimum Remaining Values)
            
            def get_candidates(tv):
                candidates = []
                for hv in range(n_vertices):
                    if hv in used_verts:
                        continue
                    valid = True
                    for nb in adj_t[tv]:
                        if mapping[nb] != -1:
                            hn = mapping[nb]
                            if not is_edge_available(hv, hn):
                                valid = False
                                break
                    if valid:
                        candidates.append(hv)
                return candidates

            def backtrack(pos):
                if time.time() > deadline:
                    return False
                
                if pos == k:
                    # Commit
                    for u, v in edges_t:
                        add_edge(mapping[u], mapping[v])
                    mappings[t_idx] = mapping[:]
                    return True
                
                # Find unmapped vertex with fewest candidates
                best_tv = -1
                best_cands = None
                min_c = float('inf')
                
                # Optimization: Only consider unmapped vertices
                for tv in range(k):
                    if mapping[tv] != -1:
                        continue
                    cands = get_candidates(tv)
                    if not cands:
                        return False
                    if len(cands) < min_c:
                        min_c = len(cands)
                        best_tv = tv
                        best_cands = cands
                        if min_c == 1:
                            break
                
                if best_tv == -1:
                    return False
                
                # Randomize candidates to explore different solutions
                local_rng.shuffle(best_cands)
                
                for hv in best_cands:
                    mapping[best_tv] = hv
                    used_verts.add(hv)
                    if backtrack(pos + 1):
                        return True
                    used_verts.remove(hv)
                    mapping[best_tv] = -1
                    
                return False
            
            return backtrack(0)

        success = True
        for t_idx in order:
            if time.time() > deadline:
                success = False
                break
            if pack_tree(t_idx):
                pass
            else:
                success = False
                break
        
        # Calculate score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])
                
        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
            
        return success

    # Try multiple seeds
    seed = 0
    while time.time() < deadline:
        try_packing(seed)
        seed += 1
        if best_score == 190:
            break
            
    # Ensure return types are correct
    # "Entry k-1 is None or a list of k distinct int host vertices"
    # "integers must be Python int"
    # "sequences must be list or tuple"
    
    final_mappings = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            final_mappings.append(None)
        else:
            # Convert to list of ints
            final_mappings.append([int(v) for v in best_mappings[i]])
            
    return final_mappings
