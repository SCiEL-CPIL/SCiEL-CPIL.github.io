import time
from random import Random

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
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    result = [None] * 20
    available = set(all_edges)
    
    # Precompute tree structures for faster access
    tree_info = []
    for t_idx in range(20):
        k = t_idx + 1
        if k <= 2:
            tree_info.append(None)
            continue
        tree_edges = trees[t_idx]
        adj = [[] for _ in range(k)]
        for (u, v) in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        # Find leaves
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        if not leaves:
            leaves = [0]
        # Compute degree for each node
        deg = [len(adj[v]) for v in range(k)]
        # Sort leaves: prefer leaves whose parent has higher degree (more constrained)
        leaf_parent_deg = []
        for lf in leaves:
            for nb in adj[lf]:
                leaf_parent_deg.append((deg[nb], lf))
                break
        leaf_parent_deg.sort(reverse=True)
        leaves = [lf for _, lf in leaf_parent_deg]
        tree_info.append((adj, leaves, tree_edges, deg))
    
    # Compute a better ordering: sort by (size descending, max degree descending, num leaves ascending)
    order_scores = []
    for t_idx in range(20):
        k = t_idx + 1
        if k <= 2:
            order_scores.append((k, 0, 0))
        else:
            info = tree_info[t_idx]
            adj, leaves, tree_edges, deg = info
            max_deg = max(deg)
            order_scores.append((k, max_deg, -len(leaves)))
    order = sorted(range(20), key=lambda i: order_scores[i], reverse=True)
    
    # Maintain available adjacency incrementally
    avail_adj = [set() for _ in range(n)]
    for (a, b) in available:
        avail_adj[a].add(b)
        avail_adj[b].add(a)
    
    def can_embed(tree_idx, avail_adj_ref, available_edges):
        """Try to find embedding for tree tree_idx in available_edges.
        Returns the embedding (list of k host vertices) or None."""
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for (a, b) in available_edges:
                return [a, b]
            return None
        
        info = tree_info[tree_idx]
        adj, leaves, tree_edges, deg = info
        
        # Try each leaf as root (already sorted by constraint)
        for root in leaves:
            # Build BFS order from root
            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            visited[root] = True
            queue = [root]
            order_list = [root]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        queue.append(v)
                        order_list.append(v)
            
            host = [-1] * k
            used_mask = 0
            
            # Precompute children count for each node in order
            child_count = [len(children[v]) for v in range(k)]
            
            def backtrack(idx):
                nonlocal used_mask
                
                if time.time() > deadline:
                    return None
                
                if idx == len(order_list):
                    # Verify all edges are in available
                    for (u, v) in tree_edges:
                        hu, hv = host[u], host[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e not in available_edges:
                            return None
                    return list(host)
                
                v = order_list[idx]
                
                if parent[v] == -1:
                    # Root: any unused vertex
                    candidates = [h for h in range(n) if not (used_mask >> h) & 1]
                else:
                    pv = parent[v]
                    if host[pv] == -1:
                        return None
                    ph = host[pv]
                    # Must be an unused neighbor of ph in available graph
                    candidates = [h for h in avail_adj_ref[ph] if not (used_mask >> h) & 1]
                
                if not candidates:
                    return None
                
                # Heuristic: sort by number of available unused neighbors (most constrained first)
                # Also consider: prefer hosts with fewer unused neighbors (more constrained)
                def score_cand(h):
                    cnt = 0
                    for nb in avail_adj_ref[h]:
                        if not (used_mask >> nb) & 1:
                            cnt += 1
                    return cnt
                
                candidates.sort(key=score_cand)
                
                for h in candidates:
                    host[v] = h
                    used_mask |= (1 << h)
                    
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    
                    host[v] = -1
                    used_mask &= ~(1 << h)
                
                return None
            
            res = backtrack(0)
            if res is not None:
                return res
        
        return None
    
    success = True
    for t_idx in order:
        if time.time() > deadline:
            break
            
        k = t_idx + 1
        if k <= 2:
            if k == 1:
                result[t_idx] = [0]
            else:
                found = None
                for (a, b) in available:
                    found = [a, b]
                    break
                if found:
                    result[t_idx] = found
                    e = (min(found[0], found[1]), max(found[0], found[1]))
                    available.discard(e)
                    avail_adj[found[0]].discard(found[1])
                    avail_adj[found[1]].discard(found[0])
                else:
                    success = False
                    break
            continue
        
        emb = can_embed(t_idx, avail_adj, available)
        if emb is None:
            success = False
            break
        
        result[t_idx] = emb
        for (u, v) in trees[t_idx]:
            hu, hv = emb[u], emb[v]
            e = (min(hu, hv), max(hu, hv))
            if e in available:
                available.discard(e)
                avail_adj[hu].discard(hv)
                avail_adj[hv].discard(hu)
            else:
                success = False
                break
        if not success:
            break
    
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result
