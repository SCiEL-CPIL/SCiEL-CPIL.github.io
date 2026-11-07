from random import Random
import time

def instance_trees(n=20, seed=611):
    rng = Random(seed)
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
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    
    adj = []
    for i, edges in enumerate(trees):
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        adj.append(adj_list)
    
    best_score = 0
    best_result = [None] * 20
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        rng = Random(attempt * 7919 + 104729)
        used_edges = set()
        embeddings = [None] * 20
        
        order = list(range(20))
        if attempt % 4 == 0:
            pass
        elif attempt % 4 == 1:
            order.reverse()
        elif attempt % 4 == 2:
            rng.shuffle(order)
        else:
            # Alternate: small, large, small, large...
            small = [i for i in range(20) if i < 10]
            large = [i for i in range(20) if i >= 10]
            mixed = []
            for j in range(10):
                mixed.append(small[j])
                mixed.append(large[j])
            order = mixed
        
        success = True
        for idx in order:
            if time.time() > deadline - 5:
                success = False
                break
            k = idx + 1
            if k == 1:
                embeddings[idx] = [0]
                continue
            
            adj_list = adj[idx]
            host = [-1] * k
            host_used = [False] * 20
            
            vert_order = list(range(k))
            vert_order.sort(key=lambda v: len(adj_list[v]), reverse=True)
            
            def backtrack(pos):
                if time.time() > deadline - 5:
                    return False
                if pos == k:
                    return True
                
                v = vert_order[pos]
                candidates = [hv for hv in range(20) if not host_used[hv]]
                rng.shuffle(candidates)
                
                for hv in candidates:
                    conflict = False
                    for nb in adj_list[v]:
                        if host[nb] != -1:
                            e = (min(hv, host[nb]), max(hv, host[nb]))
                            if e in used_edges:
                                conflict = True
                                break
                    if conflict:
                        continue
                    
                    host[v] = hv
                    host_used[hv] = True
                    if backtrack(pos + 1):
                        return True
                    host[v] = -1
                    host_used[hv] = False
                
                return False
            
            if backtrack(0):
                embeddings[idx] = [host[i] for i in range(k)]
                for u, v in trees[idx]:
                    e = (min(host[u], host[v]), max(host[u], host[v]))
                    used_edges.add(e)
            else:
                success = False
                break
        
        if success and all(embeddings[i] is not None for i in range(20)):
            return [list(embeddings[i]) for i in range(20)]
        
        partial_score = sum(i for i in range(20) if embeddings[i] is not None)
        if partial_score > best_score:
            best_score = partial_score
            best_result = [list(embeddings[i]) if embeddings[i] is not None else None for i in range(20)]
    
    return best_result
