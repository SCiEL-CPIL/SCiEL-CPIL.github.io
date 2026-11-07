import time
from random import Random
from collections import deque

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
    deadline = start_time + 130.0
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute tree info
    tree_info = []
    for i, tree_edges in enumerate(trees):
        k = len(tree_edges) + 1
        # Build adjacency
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        # BFS order from 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        tree_info.append({
            'k': k,
            'edges': tree_edges,
            'adj': adj,
            'parent': parent,
            'order': order
        })
    
    # Sort by size descending for packing priority
    tree_indices = list(range(n))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    def find_embedding(tree_idx, used_edges, rng, max_backtracks=50000):
        """
        Finds an injective embedding of tree into K20 using edges not in used_edges.
        Returns list of length k where result[i] is host vertex for tree vertex i, or None.
        """
        info = tree_info[tree_idx]
        k = info['k']
        if k == 1:
            # Just pick any unused vertex
            for v in range(n):
                if v not in used_edges:  # For k=1, no edges used
                    return [v]
            return None
        
        parent = info['parent']
        order = info['order']
        edges = info['edges']
        
        host = [-1] * k
        used_hosts = set()
        backtracks = [0]
        
        def backtrack(idx):
            if idx == k:
                return True
            
            if backtracks[0] > max_backtracks:
                return False
            
            tv = order[idx]
            p_tv = parent[tv]
            
            # Generate candidates
            candidates = list(range(n))
            rng.shuffle(candidates)
            
            for hv in candidates:
                if hv in used_hosts:
                    continue
                
                if p_tv != -1:
                    p_hv = host[p_tv]
                    e = (min(hv, p_hv), max(hv, p_hv))
                    if e in used_edges:
                        continue
                
                host[tv] = hv
                used_hosts.add(hv)
                
                if backtrack(idx + 1):
                    return True
                
                used_hosts.remove(hv)
                host[tv] = -1
                backtracks[0] += 1
            
            return False
        
        if backtrack(0):
            return host[:]
        else:
            return None
    
    def try_pack(seed, max_backtracks_per_tree=20000):
        rng = Random(seed)
        used = set()
        res = [None] * n
        score = 0
        
        for idx in tree_indices:
            if time.time() > deadline:
                break
            
            embedding = find_embedding(idx, used, rng, max_backtracks_per_tree)
            
            if embedding is not None:
                res[idx] = embedding
                score += len(trees[idx])
                for u, v in trees[idx]:
                    hu = embedding[u]
                    hv = embedding[v]
                    e = (min(hu, hv), max(hv))
                    used.add(e)
                    
        return res, score
    
    def try_pack_systematic():
        """Try a more systematic approach with better ordering."""
        used = set()
        res = [None] * n
        score = 0
        rng = Random(42)
        
        for idx in tree_indices:
            if time.time() > deadline:
                break
            
            # Try multiple random seeds for this tree
            found = False
            for attempt_seed in range(50):
                if time.time() > deadline:
                    break
                arng = Random(attempt_seed * 1000 + idx)
                embedding = find_embedding(idx, used, arng, 10000)
                if embedding is not None:
                    res[idx] = embedding
                    score += len(trees[idx])
                    for u, v in trees[idx]:
                        hu = embedding[u]
                        hv = embedding[v]
                        e = (min(hu, hv), max(hv))
                        used.add(e)
                    found = True
                    break
            
            if not found:
                # Try with different vertex ordering
                pass
        
        return res, score
    
    best_res = None
    best_score = 0
    
    # Try multiple seeds
    for seed in range(500):
        if time.time() > deadline:
            break
        
        res, score = try_pack(seed, max_backtracks_per_tree=15000)
        
        if score > best_score:
            best_score = score
            best_res = res
            
        if best_score == 190:
            break
    
    # If not perfect, try systematic approach
    if best_score < 190:
        res, score = try_pack_systematic()
        if score > best_score:
            best_score = score
            best_res = res
    
    # Fallback: ensure at least valid partial result
    if best_res is None:
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback
    
    # Ensure types are correct: Python int, list
    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])
    
    return final_result
