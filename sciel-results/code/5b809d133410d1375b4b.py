from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
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
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i, j))
    
    # We need to assign each tree's edges to distinct edges in K20
    # Result: result[k-1] is None or list of k vertex images
    
    # Process trees from largest to smallest (more constrained first)
    order = list(range(1, n+1))  # k=1 to 20
    order.reverse()  # 20, 19, ..., 1
    
    used_edges = set()
    assignment = [None] * n  # assignment[k-1] = list of vertex images or None
    
    # For backtracking, we need to try different embeddings
    # This could be expensive, so let's try a greedy approach first
    
    def try_embed(tree_idx, edges_list, available_vertices):
        """Try to embed tree with edges_list using available_vertices"""
        k = len(available_vertices)
        if k == 0:
            return []
        if k == 1:
            return [available_vertices[0]]
        
        # Build adjacency for the tree
        adj = {i: [] for i in range(k)}
        for u, v in edges_list:
            adj[u].append(v)
            adj[v].append(u)
        
        # Try all permutations? Too expensive. Use backtracking.
        # Pick a root, then assign neighbors.
        # Use DFS-based backtracking
        
        mapping = {}  # tree vertex -> host vertex
        used_hosts = set()
        
        def backtrack(vertex):
            if vertex is not None:
                if vertex in mapping:
                    return True
                for host in available_vertices:
                    if host in used_hosts:
                        continue
                    # Check if assigning this host works for already-assigned neighbors
                    valid = True
                    for nb in adj[vertex]:
                        if nb in mapping:
                            edge = (min(vertex, nb), max(vertex, nb))
                            host_edge = (min(host, mapping[nb]), max(host, mapping[nb]))
                            if host_edge in used_edges or host_edge in [ (min(mapping[a], mapping[b]), max(mapping[a], mapping[b])) for a,b in []]:
                                # Need to check against all currently used edges in this embedding
                                pass
                    mapping[vertex] = host
                    used_hosts.add(host)
                    
                    # Try to assign neighbors
                    for nb in adj[vertex]:
                        if nb not in mapping:
                            if not backtrack(nb):
                                break
                    else:
                        return True
                    
                    # Backtrack
                    del mapping[vertex]
                    used_hosts.discard(host)
                    return False
            else:
                # All assigned
                return True
            
            return False
        
        # Start from vertex 0
        if not backtrack(0):
            return None
        
        # Verify all edges are valid and unused
        for u, v in edges_list:
            host_u = mapping[u]
            host_v = mapping[v]
            host_edge = (min(host_u, host_v), max(host_u, host_v))
            if host_edge in used_edges:
                return None
        
        return [mapping[i] for i in range(k)]
    
    # This approach is getting complex. Let me try a different strategy.
    # Use a simple greedy with randomization and restarts.
    
    import random
    
    best_result = None
    best_score = -1
    
    for trial in range(100):
        rng = Random(trial)
        used_edges = set()
        assignment = [None] * n
        success = True
        
        # Process in random order or largest first
        order = list(range(1, n+1))
        rng.shuffle(order)
        
        for k in order:
            if k == 1:
                assignment[0] = None
                continue
            edges_list = trees[k-1]
            
            # Try to find an embedding
            found = False
            attempts = 0
            while not found and attempts < 50:
                attempts += 1
                # Pick random subset of k vertices
                verts = rng.sample(range(n), k)
                # Try random permutation
                perm = list(range(k))
                rng.shuffle(perm)
                mapping = {}
                valid = True
                for i in range(k):
                    mapping[i] = verts[perm[i]]
                
                # Check all edges
                for u, v in edges_list:
                    hu = mapping[u]
                    hv = mapping[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        valid = False
                        break
                    if hu == hv:
                        valid = False
                        break
                
                if valid:
                    # Check for duplicate vertices (shouldn't happen with sample)
                    if len(set(mapping.values())) == k:
                        for u, v in edges_list:
                            hu = mapping[u]
                            hv = mapping[v]
                            used_edges.add((min(hu, hv), max(hu, hv)))
                        assignment[k-1] = [mapping[i] for i in range(k)]
                        found = True
            
            if not found:
                success = False
                break
        
        if success:
            # Count total edges
            total = sum(len(trees[k-1]) for k in range(1, n+1) if assignment[k-1] is not None)
            if total > best_score:
                best_score = total
                best_result = assignment[:]
    
    if best_result is not None:
        return best_result
    
    # Fallback: just return the identity for T20
    return [None] * 19 + [list(range(20))]
