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
        for x in code: degree[x] += 1
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
    
    used_edges = set()
    result = [None] * 20
    
    def add_edge(u, v):
        if u > v: u, v = v, u
        used_edges.add((u, v))

    # Embed T20 (index 19)
    if time.time() < deadline:
        # Identity map
        result[19] = list(range(20))
        for u, v in trees[19]:
            add_edge(u, v)
            
    # Embed T19..T1
    for k_idx in range(18, -1, -1):
        if time.time() > deadline:
            break
            
        k = k_idx + 1
        if k == 1:
            result[0] = [0]
            continue
            
        tree_edges = trees[k_idx]
        
        # Degrees
        deg = [0] * k
        for u, v in tree_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Order vertices by degree desc
        vertices = sorted(range(k), key=lambda x: (-deg[x], x))
        
        # Neighbors
        neighbors = [[] for _ in range(k)]
        for u, v in tree_edges:
            neighbors[u].append(v)
            neighbors[v].append(u)
            
        mapping = [-1] * k
        host_used = [False] * 20
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(vertices):
                return True
                
            v = vertices[idx]
            
            # Find valid candidates
            candidates = []
            for h in range(20):
                if host_used[h]:
                    continue
                valid = True
                for nv in neighbors[v]:
                    if mapping[nv] != -1:
                        nh = mapping[nv]
                        edge = (min(h, nh), max(h, nh))
                        if edge in used_edges:
                            valid = False
                            break
                if valid:
                    candidates.append(h)
                    
            for h in candidates:
                mapping[v] = h
                host_used[h] = True
                if backtrack(idx + 1):
                    return True
                mapping[v] = -1
                host_used[h] = False
                
            return False

        if backtrack(0):
            result[k_idx] = [mapping[i] for i in range(k)]
            for u, v in tree_edges:
                add_edge(mapping[u], mapping[v])
        else:
            result[k_idx] = None

    return result
