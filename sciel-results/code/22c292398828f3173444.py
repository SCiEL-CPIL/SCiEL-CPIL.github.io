import time
import random

def instance_trees(n=20, seed=611):
    from random import Random
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 20
    trees = instance_trees(n)
    
    available_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            available_edges.add((u, v))
    
    result = [None] * n
    
    tree_adj = [None] * n
    for k in range(1, n + 1):
        if k == 1:
            tree_adj[k-1] = [[]]
        elif k == 2:
            tree_adj[k-1] = [[1], [0]]
        else:
            adj = [[] for _ in range(k)]
            for u, v in trees[k-1]:
                adj[u].append(v)
                adj[v].append(u)
            tree_adj[k-1] = adj

    def embed_tree(k, available_edges, timeout):
        if k == 1:
            return [0]
            
        if k == 2:
            for e in available_edges:
                return [e[0], e[1]]
            return None

        adj = tree_adj[k-1]
        order = sorted(range(k), key=lambda v: -len(adj[v]))
        rng = random.Random()

        def backtrack(idx, map_th, used_hosts):
            if time.time() > timeout:
                return False
            
            if idx == k:
                return True
            
            v = order[idx]
            
            mapped_neighbors = []
            for u in adj[v]:
                if u in map_th:
                    mapped_neighbors.append(map_th[u])
            
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                
                valid = True
                for mh in mapped_neighbors:
                    edge = (min(mh, h), max(mh, h))
                    if edge not in available_edges:
                        valid = False
                        break
                
                if valid:
                    candidates.append(h)
            
            rng.shuffle(candidates)
            
            for h in candidates:
                map_th[v] = h
                used_hosts.add(h)
                
                if backtrack(idx + 1, map_th, used_hosts):
                    return True
                
                del map_th[v]
                used_hosts.remove(h)
            
            return False

        for attempt in range(100):
            if time.time() > timeout:
                break
            if backtrack(0, {}, set()):
                res_list = [0] * k
                for tv, hv in map_th.items():
                    res_list[tv] = hv
                return res_list
        return None

    for k in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        embedding = embed_tree(k, available_edges, deadline)
        
        if embedding is not None:
            result[k-1] = embedding
            tree_edges = trees[k-1]
            for u, v in tree_edges:
                hu = embedding[u]
                hv = embedding[v]
                edge = (min(hu, hv), max(hu, hv))
                if edge in available_edges:
                    available_edges.remove(edge)

    return result
# EVOLVE-BLOCK-END
