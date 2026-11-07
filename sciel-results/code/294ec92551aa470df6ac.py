import time
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
    start_time = time.time()
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Process larger trees first
    order = sorted(range(n), key=lambda i: len(trees[i]), reverse=True)
    
    best_result = [None] * n
    best_edges = 0
    
    # Precompute adjacency for each tree
    tree_adj = []
    for idx in range(n):
        k = idx + 1
        adj = [[] for _ in range(k)]
        for (u, v) in trees[idx]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    def try_pack(seed_offset=0):
        nonlocal best_result, best_edges
        rng = Random(42 + seed_offset)
        result = [None] * n
        used_edges = set()
        
        for idx in order:
            k = idx + 1
            if k == 1:
                result[idx] = [0]
                continue
            if k == 2:
                placed = False
                for _ in range(2000):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a == b: continue
                    e = (min(a,b), max(a,b))
                    if e not in used_edges:
                        result[idx] = [a, b]
                        used_edges.add(e)
                        placed = True
                        break
                if not placed:
                    return False
                continue
            
            # Backtracking for larger trees
            tree_edge_set = set()
            for (u, v) in trees[idx]:
                tree_edge_set.add((u, v))
                tree_edge_set.add((v, u))
            
            placed = False
            for attempt in range(300):
                if time.time() > deadline:
                    return False
                mapping = [-1] * k
                mapping_used = [False] * 20
                
                def backtrack(pos):
                    if time.time() > deadline:
                        return False
                    if pos == k:
                        return True
                    
                    candidates = list(range(20))
                    rng.shuffle(candidates)
                    
                    for host in candidates:
                        if mapping_used[host]:
                            continue
                        
                        # Check edges to already-mapped vertices
                        ok = True
                        for j in range(pos):
                            if (pos, j) in tree_edge_set:
                                e = (min(host, mapping[j]), max(host, mapping[j]))
                                if e in used_edges:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        
                        mapping[pos] = host
                        mapping_used[host] = True
                        
                        # Temporarily add edges
                        new_edges = set()
                        for j in range(pos):
                            if (pos, j) in tree_edge_set:
                                e = (min(host, mapping[j]), max(host, mapping[j]))
                                new_edges.add(e)
                        for e in new_edges:
                            used_edges.add(e)
                        
                        if backtrack(pos + 1):
                            return True
                        
                        for e in new_edges:
                            used_edges.discard(e)
                        mapping_used[host] = False
                        mapping[pos] = -1
                    
                    return False
                
                if backtrack(0):
                    result[idx] = mapping[:]
                    placed = True
                    break
            
            if not placed:
                return False
        
        total = sum(len(trees[i]) for i in range(n) if result[i] is not None)
        if total > best_edges:
            best_edges = total
            best_result = [list(r) if r is not None else None for r in result]
        return True
    
    for s in range(2000):
        if time.time() > deadline:
            break
        try_pack(s)
    
    return best_result
