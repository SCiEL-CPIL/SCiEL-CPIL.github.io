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
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20
    
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
    
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda x: -(x + 1))
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def count_valid_embeddings(t_idx, used_edges_set, used_verts_set):
        """Count how many valid embeddings exist for tree t_idx"""
        k = t_idx + 1
        if k == 1:
            return n_vertices - len(used_verts_set)
        adj_t = tree_adj[t_idx]
        edges_t = tree_edges[t_idx]
        
        # Use backtracking to count (with a limit)
        count = 0
        mapping = [-1] * k
        
        def count_backtrack(pos):
            nonlocal count
            if count > 100:  # Limit counting
                return
            if pos == k:
                count += 1
                return
            for tv in range(k):
                if mapping[tv] != -1:
                    continue
                for hv in range(n_vertices):
                    if hv in used_verts_set:
                        continue
                    valid = True
                    for nb in adj_t[tv]:
                        if mapping[nb] != -1:
                            hn = mapping[nb]
                            e = (min(hv, hn), max(hv, hn))
                            if e in used_edges_set:
                                valid = False
                                break
                    if valid:
                        mapping[tv] = hv
                        count_backtrack(pos + 1)
                        mapping[tv] = -1
                break  # Only try one vertex per step for speed
        
        # This is too complex. Let's just return a simple estimate.
        # Estimate based on available vertices and edges
        avail_verts = n_vertices - len(used_verts_set)
        if avail_verts < k:
            return 0
        # Rough estimate
        return avail_verts * (avail_verts - 1) // 2  # Very rough
    
    def try_packing(attempt_seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * n_trees
        
        for order_idx in range(n_trees):
            if time.time() > deadline:
                break
            
            t_idx = tree_indices[order_idx]
            k = t_idx + 1
            adj_t = tree_adj[t_idx]
            edges_t = tree_edges[t_idx]
            
            if k == 1:
                mappings[t_idx] = [0]
                continue
            
            mapping = [-1] * k
            used_verts = set()
            
            def get_candidates(tv):
                candidates = []
                for hv in range(n_vertices):
                    if hv in used_verts:
                        continue
                    valid = True
                    for nb in adj_t[tv]:
                        if mapping[nb] != -1:
                            hn = mapping[nb]
                            e = (min(hv, hn), max(hv, hn))
                            if e in used_edges:
                                valid = False
                                break
                    if valid:
                        candidates.append(hv)
                return candidates
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                
                if pos == k:
                    for u, v in edges_t:
                        hu, hv = mapping[u], mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    mappings[t_idx] = mapping[:]
                    return True
                
                best_tv = -1
                best_candidates = None
                min_cand_count = float('inf')
                
                for tv in range(k):
                    if mapping[tv] != -1:
                        continue
                    candidates = get_candidates(tv)
                    if not candidates:
                        return False
                    if len(candidates) < min_cand_count:
                        min_cand_count = len(candidates)
                        best_tv = tv
                        best_candidates = candidates
                        if min_cand_count == 1:
                            break
                
                if best_tv == -1:
                    return False
                
                # Randomize order slightly
                local_rng.shuffle(best_candidates)
                
                for hv in best_candidates:
                    mapping[best_tv] = hv
                    used_verts.add(hv)
                    if backtrack(pos + 1):
                        return True
                    used_verts.remove(hv)
                    mapping[best_tv] = -1
                
                return False
            
            if not backtrack(0):
                break
        
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])
        
        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
        
        return best_score == 190
    
    attempt = 0
    while time.time() < deadline:
        if try_packing(attempt):
            break
        attempt += 1
    
    result = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    
    return result
