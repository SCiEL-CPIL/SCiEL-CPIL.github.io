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
    deadline = start_time + 145.0
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Sort tree indices by size (descending) to pack larger trees first
    tree_indices = list(range(n))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    def find_embedding_random(tree_edges, used_edges_global, rng):
        """
        Finds an injective embedding of tree_edges into K20 
        using edges not in used_edges_global.
        Returns a list of length k where result[i] is host vertex for tree vertex i, or None.
        """
        k = len(tree_edges) + 1
        if k == 1:
            return [0]
        
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS order from 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
                    
        host = [-1] * k
        used_hosts = set()
        
        def backtrack(idx):
            if idx == k:
                return True
            
            tv = order[idx]
            candidates = list(range(n))
            rng.shuffle(candidates)
            
            p_tv = parent[tv]
            
            for hv in candidates:
                if hv in used_hosts:
                    continue
                
                if p_tv != -1:
                    p_hv = host[p_tv]
                    e = (min(hv, p_hv), max(hv, p_hv))
                    if e in used_edges_global:
                        continue
                
                host[tv] = hv
                used_hosts.add(hv)
                
                if backtrack(idx + 1):
                    return True
                
                used_hosts.remove(hv)
                host[tv] = -1
            
            return False
        
        if backtrack(0):
            return host[:]
        else:
            return None

    def try_pack_greedy(seed):
        rng = Random(seed)
        used = set()
        res = [None] * n
        score = 0
        
        for idx in tree_indices:
            if time.time() > deadline:
                break
                
            tree = trees[idx]
            embedding = find_embedding_random(tree, used, rng)
            
            if embedding is not None:
                res[idx] = embedding
                score += len(tree)
                for u, v in tree:
                    hu = embedding[u]
                    hv = embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    used.add(e)
                    
        return res, score

    best_res = None
    best_score = 0
    
    for seed in range(300):
        if time.time() > deadline:
            break
            
        res, score = try_pack_greedy(seed)
        
        if score > best_score:
            best_score = score
            best_res = res
            
        if best_score == 190:
            break
            
    if best_res is None:
        # Fallback: embed T1 and T2 at least
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback

    # Ensure types are correct
    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])
            
    return final_result
