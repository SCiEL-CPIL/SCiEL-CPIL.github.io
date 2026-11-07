import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # Generate trees
    rng = Random(611)
    n = 20
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
        
    # Order: Largest first
    order = list(range(n-1, -1, -1))
    
    all_edges = set((i, j) for i in range(n) for j in range(i+1, n))
    used_edges = set()
    embeddings = [None] * n
    
    def backtrack(pos):
        if pos == len(order):
            return True
        if time.time() > deadline:
            return False
            
        t_idx = order[pos]
        edges = trees[t_idx]
        k = len(edges) + 1 if edges else 1
        
        if k == 1:
            embeddings[t_idx] = [0]
            return backtrack(pos + 1)
            
        if k == 2:
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) not in used_edges:
                        embeddings[t_idx] = [u, v]
                        used_edges.add((u, v))
                        if backtrack(pos + 1):
                            return True
                        used_edges.remove((u, v))
            return False
            
        # k >= 3
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        def embed(mapping):
            # Find next unassigned tree vertex
            next_v = -1
            for v in range(k):
                if v not in mapping:
                    if any(n in mapping for n in adj[v]):
                        next_v = v
                        break
            if next_v == -1:
                # All assigned
                for u, v in edges:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        return False
                final_map = [0]*k
                for tv, hv in mapping.items():
                    final_map[tv] = hv
                embeddings[t_idx] = final_map
                for u, v in edges:
                    hu, hv = final_map[u], final_map[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
                
                if backtrack(pos + 1):
                    return True
                    
                for u, v in edges:
                    hu, hv = final_map[u], final_map[v]
                    used_edges.remove((min(hu, hv), max(hu, hv)))
                return False
            
            for hv in range(n):
                if hv in mapping.values():
                    continue
                ok = True
                for n_ in adj[next_v]:
                    if n_ in mapping:
                        hn = mapping[n_]
                        e = (min(hv, hn), max(hv, hn))
                        if e in used_edges:
                            ok = False
                            break
                if not ok:
                    continue
                mapping[next_v] = hv
                if embed(mapping):
                    return True
                del mapping[next_v]
            return False
            
        if pos == 0:
            mapping = {0: 0}
            if embed(mapping):
                return True
        else:
            for hv in range(n):
                mapping = {0: hv}
                if embed(mapping):
                    return True
        return False

    if backtrack(0):
        return embeddings
    else:
        return embeddings

