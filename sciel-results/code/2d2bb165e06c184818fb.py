import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Pre-compute K20 edge bitmasks
    # Edge (i, j) -> bit index i * 20 + j (if i < j) or similar
    # Simpler: set of tuples is fine for 190 edges.
    # But bitset is faster.
    # Let's stick to set for clarity, optimize if needed.
    
    # Process trees in reverse order (largest first)
    order = list(range(n - 1, -1, -1))
    
    used_edges = set()
    result = [None] * n
    
    # Adjacency of K20
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            adj[i][j] = True
            adj[j][i] = True
            
    def try_embed(tree_idx, tree_edges, k, used_edges_set):
        if time.time() > deadline:
            return None
            
        # Build tree adjacency
        tree_adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
            
        host_map = [-1] * k
        assigned = [False] * k
        
        # Heuristic: Start with vertex 0? Or vertex with max degree?
        # Let's start with vertex 0.
        # Better: Start with a vertex that has high degree in the tree.
        start_v = 0
        max_deg = 0
        for i in range(k):
            if len(tree_adj[i]) > max_deg:
                max_deg = len(tree_adj[i])
                start_v = i
                
        host_map[start_v] = 0 # Symmetry breaking: fix one vertex to 0
        assigned[start_v] = True
        # Note: This fixes the rotation/reflection of the tree in the host.
        # Is this valid? Yes, K20 is vertex-transitive.
        
        # Recursive backtracking
        def backtrack(depth):
            if time.time() > deadline:
                return False
                
            if depth == k:
                return True
                
            # Find next vertex to assign
            # Priority: Unassigned vertices with most assigned neighbors
            best_v = -1
            max_assigned_neighbors = -1
            for v in range(k):
                if not assigned[v]:
                    cnt = 0
                    for nb in tree_adj[v]:
                        if assigned[nb]:
                            cnt += 1
                    if cnt > max_assigned_neighbors:
                        max_assigned_neighbors = cnt
                        best_v = v
                        
            if best_v == -1:
                return False # Should not happen if depth < k
                
            # Candidates for best_v
            candidates = []
            for h in range(n):
                if host_map.index(h) != -1: # h is used by another vertex in this tree
                    continue
                # Check edges to assigned neighbors
                valid = True
                for nb in tree_adj[best_v]:
                    if assigned[nb]:
                        nb_host = host_map[nb]
                        edge = (min(h, nb_host), max(h, nb_host))
                        if edge in used_edges_set:
                            valid = False
                            break
                if valid:
                    candidates.append(h)
                    
            # Order candidates: Least constraining?
            # Or just try all.
            # To speed up, maybe shuffle with a fixed seed?
            # Deterministic order: by index.
            
            for h in candidates:
                host_map[best_v] = h
                assigned[best_v] = True
                # Add edges to used_edges_set? 
                # We need to add edges to used_edges_set globally?
                # No, used_edges_set is passed by reference (mutable).
                # We must add edges from best_v to assigned neighbors.
                new_edges = []
                for nb in tree_adj[best_v]:
                    if assigned[nb]:
                        nb_host = host_map[nb]
                        edge = (min(h, nb_host), max(h, nb_host))
                        used_edges_set.add(edge)
                        new_edges.append(edge)
                        
                if backtrack(depth + 1):
                    return True
                    
                # Undo
                for e in new_edges:
                    used_edges_set.discard(e)
                host_map[best_v] = -1
                assigned[best_v] = False
                
            return False
            
        if backtrack(1): # depth 1 because start_v is assigned
            return list(host_map)
        return None

    for tree_idx in order:
        if time.time() > deadline:
            break
            
        k = tree_idx + 1
        tree_edges = trees[tree_idx]
        
        if k == 1:
            result[tree_idx] = [0]
            continue
            
        if k == 2:
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in used_edges:
                        result[tree_idx] = [i, j]
                        used_edges.add((i, j))
                        found = True
                        break
                if found:
                    break
            if not found:
                result[tree_idx] = None
            continue
            
        res = try_embed(tree_idx, tree_edges, k, used_edges)
        if res is not None:
            result[tree_idx] = res
            for (a, b) in tree_edges:
                ha, hb = res[a], res[b]
                edge = (min(ha, hb), max(ha, hb))
                used_edges.add(edge)
        else:
            result[tree_idx] = None
            
    # Final check
    total_edges = 0
    for i in range(n):
        if result[i] is not None:
            total_edges += len(trees[i])
            
    return result
