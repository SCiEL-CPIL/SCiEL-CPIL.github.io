import time
import random
from random import Random
from collections import deque

def _generate_trees(n=20, seed=611):
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
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    trees = _generate_trees(20, 611)
    n = 20
    
    # Precompute tree properties
    tree_info = []
    for t_idx in range(20):
        k = t_idx + 1
        edges = trees[t_idx]
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'leaves': [0], 'max_deg': 0})
        elif k == 2:
            tree_info.append({'k': 2, 'edges': edges, 'leaves': [0, 1], 'max_deg': 1})
        else:
            adj = [[] for _ in range(k)]
            for (u, v) in edges:
                adj[u].append(v)
                adj[v].append(u)
            leaves = [v for v in range(k) if len(adj[v]) == 1]
            max_deg = max(len(adj[v]) for v in range(k))
            tree_info.append({'k': k, 'edges': edges, 'leaves': leaves, 'max_deg': max_deg, 'adj': adj})
    
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    def embed_tree(t_idx, available_edges, rng, time_limit=3.0):
        """Embed tree t_idx into available_edges. Returns list of host vertices or None."""
        info = tree_info[t_idx]
        k = info['k']
        
        if k == 1:
            return [0]
        if k == 2:
            if not available_edges:
                return None
            (a, b) = next(iter(available_edges))
            return [a, b]
        
        adj = info['adj']
        leaves = info['leaves']
        edges = info['edges']
        
        # Build host adjacency
        host_adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            host_adj[a].add(b)
            host_adj[b].add(a)
        
        # Try different roots
        roots = list(leaves)
        rng.shuffle(roots)
        
        for root in roots:
            if time.time() > start_time + time_limit:
                continue
            
            # BFS to get parent-child structure
            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            visited[root] = True
            queue = deque([root])
            order = [root]
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        queue.append(v)
                        order.append(v)
            
            # Shuffle children for randomization
            for v in range(k):
                rng.shuffle(children[v])
            
            host = [-1] * k
            used = set()
            
            # Forward checking: track available neighbors for each used host
            avail_neigh = {i: set(host_adj[i]) for i in range(n)}
            
            def backtrack(idx):
                if time.time() > start_time + time_limit:
                    return None
                
                if idx == len(order):
                    # Verify all edges
                    for (u, v) in edges:
                        hu, hv = host[u], host[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e not in available_edges:
                            return None
                    return list(host)
                
                v = order[idx]
                
                if parent[v] == -1:
                    candidates = list(range(n))
                else:
                    pv = parent[v]
                    if host[pv] == -1:
                        return None
                    candidates = list(avail_neigh[host[pv]])
                
                # Filter: not used
                valid = [h for h in candidates if h not in used]
                
                if not valid:
                    return None
                
                # Heuristic: prefer hosts with fewer available neighbors (more constrained)
                valid.sort(key=lambda h: len(avail_neigh[h]))
                
                for h in valid:
                    host[v] = h
                    used.add(h)
                    
                    # Update available neighbors
                    for nb in host_adj[h]:
                        if nb in avail_neigh:
                            avail_neigh[nb].discard(h)
                    avail_neigh[h] = set()
                    
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    
                    # Undo
                    host[v] = -1
                    used.discard(h)
                    avail_neigh[h] = set(host_adj[h]) - used
                    for nb in host_adj[h]:
                        if nb in avail_neigh:
                            avail_neigh[nb].add(h)
                
                return None
            
            result = backtrack(0)
            if result is not None:
                return result
        
        return None
    
    # Try multiple strategies
    best_result = None
    best_score = -1
    
    # Strategy 1: Process in decreasing size
    order1 = list(range(19, -1, -1))
    
    # Strategy 2: Process in increasing size
    order2 = list(range(20))
    
    # Strategy 3: Interleave small and large
    order3 = [19, 0, 18, 1, 17, 2, 16, 3, 15, 4, 14, 5, 13, 6, 12, 7, 11, 8, 10, 9]
    
    strategies = [order1, order2, order3]
    
    for strategy_idx, order in enumerate(strategies):
        if time.time() > deadline - 10:
            break
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = Random(restart * 1000 + strategy_idx * 100 + 42)
            result = [None] * 20
            available = set(all_edges)
            success = True
            
            for t_idx in order:
                if time.time() > deadline - 3:
                    success = False
                    break
                
                k = t_idx + 1
                if k == 1:
                    result[t_idx] = [0]
                    continue
                elif k == 2:
                    if not available:
                        success = False
                        break
                    (a, b) = rng.choice(list(available))
                    result[t_idx] = [a, b]
                    available.discard((min(a, b), max(a, b)))
                    continue
                
                time_limit = min(4.0, max(0.5, (deadline - 3 - time.time()) * 0.2))
                emb = embed_tree(t_idx, available, rng, time_limit=time_limit)
                if emb is None:
                    success = False
                    break
                
                result[t_idx] = emb
                for (u, v) in trees[t_idx]:
                    hu, hv = emb[u], emb[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in available:
                        available.discard(e)
                    else:
                        success = False
                        break
                if not success:
                    break
            
            if success:
                # Verify
                used_edges = set()
                valid = True
                for t_idx in range(20):
                    if result[t_idx] is None:
                        valid = False
                        break
                    emb = result[t_idx]
                    if len(emb) != t_idx + 1 or len(set(emb)) != t_idx + 1:
                        valid = False
                        break
                    for x in emb:
                        if not isinstance(x, int) or x < 0 or x >= 20:
                            valid = False
                            break
                    if not valid:
                        break
                    for (u, v) in trees[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            valid = False
                            break
                        used_edges.add(e)
                    if not valid:
                        break
                
                if valid:
                    score = len(used_edges)
                    if score > best_score:
                        best_score = score
                        best_result = [list(result[i]) for i in range(20)]
                    if score == 190:
                        for i in range(20):
                            best_result[i] = [int(x) for x in best_result[i]]
                        return best_result
    
    if best_result is not None:
        for i in range(20):
            if best_result[i] is not None:
                best_result[i] = [int(x) for x in best_result[i]]
        return best_result
    
    # Fallback: return empty
    return [None] * 20
