import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            if k == 1:
                trees.append([])
            else:
                trees.append([(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
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
    deadline = start_time + 115.0
    
    trees = instance_trees(20, 611)
    
    # Precompute adjacency and roots
    tree_data = []
    for i, t in enumerate(trees):
        k = len(t) + 1 if len(t) > 0 else 1
        if k == 1:
            tree_data.append((0, i, [], None)) # size, idx, edges, adj
            continue
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        root = max(range(k), key=lambda x: len(adj[x]))
        tree_data.append((len(t), i, t, adj, root))
        
    # Sort by size descending
    tree_data.sort(key=lambda x: x[0], reverse=True)
    
    all_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            all_edges.add((u, v))
            
    best_result = [None] * 20
    best_score = -1
    
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        current_remaining = set(all_edges)
        current_result = [None] * 20
        current_score = 0
        rng = Random(restart)
        
        for item in tree_data:
            if time.time() > deadline:
                break
                
            size, orig_idx, tree_edges, adj, root = item
            k = size + 1 if size > 0 else 1
            
            if k == 1:
                current_result[orig_idx] = [0]
                continue
            
            found_embedding = False
            attempts = 5
            
            for _ in range(attempts):
                root_host = rng.randint(0, 19)
                
                mapping = [-1] * k
                mapping[root] = root_host
                visited_host = {root_host}
                assigned_count = 1
                
                def dfs(node, parent):
                    nonlocal assigned_count
                    if time.time() > deadline:
                        return False
                    if assigned_count == k:
                        return True
                    
                    neighbors = [n for n in adj[node] if mapping[n] == -1]
                    neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                    
                    for n in neighbors:
                        cur_host = mapping[node]
                        candidates = []
                        for h in range(20):
                            if h not in visited_host:
                                e = (min(cur_host, h), max(cur_host, h))
                                if e in current_remaining:
                                    candidates.append(h)
                        
                        rng.shuffle(candidates)
                        
                        for h in candidates:
                            mapping[n] = h
                            visited_host.add(h)
                            assigned_count += 1
                            if dfs(n, node):
                                return True
                            assigned_count -= 1
                            mapping[n] = -1
                            visited_host.remove(h)
                    return False
                
                if dfs(root, -1):
                    for u, v in tree_edges:
                        e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                        current_remaining.remove(e)
                    current_result[orig_idx] = mapping
                    current_score += size
                    found_embedding = True
                    break
            
            if not found_embedding:
                current_result[orig_idx] = None
        
        if current_score > best_score:
            best_score = current_score
            best_result = current_result[:]
            
        if best_score == 190:
            break
            
    return best_result
