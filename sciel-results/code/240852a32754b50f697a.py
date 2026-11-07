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
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    
    # Precompute K20 edges
    K20_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            K20_edges.add((u, v))
            
    def solve_packing(trees, deadline):
        """
        Attempts to find a perfect edge-disjoint packing of the trees in K20.
        Returns a list of 20 embeddings (lists of host vertices) or None.
        """
        best_embeddings = None
        best_score = 0
        
        # Number of restarts to try
        num_restarts = 200
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break
                
            rng = Random(1000 + restart * 7 + 13)
            used_edges = set()
            embeddings = [None] * 20
            
            # Embed T20 first
            t20_edges = trees[19]
            t20_adj = [[] for _ in range(20)]
            for u, v in t20_edges:
                t20_adj[u].append(v)
                t20_adj[v].append(u)
                
            # Root T20 at 0
            parent = [-1] * 20
            order = [0]
            visited = [False] * 20
            visited[0] = True
            queue = [0]
            while queue:
                u = queue.pop(0)
                for v in t20_adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        queue.append(v)
                        order.append(v)
                        
            # Backtracking for T20 with better heuristics
            host20 = [None] * 20
            host20[0] = 0
            used_hosts = {0}
            
            # Precompute children for each node in the rooted tree
            children20 = [[] for _ in range(20)]
            for v in range(1, 20):
                children20[parent[v]].append(v)
            
            # Order nodes by depth (BFS order already gives this)
            # For better pruning, process nodes with more constraints first
            # Within same level, process nodes whose parent has fewer available hosts first
            
            def backtrack_t20(idx):
                if time.time() > deadline - 1:
                    return False
                if idx == 20:
                    return True
                
                v = order[idx]
                p = parent[v]
                hp = host20[p]
                
                # Candidates for v: host vertices not in used_hosts
                # such that (hp, h) is not in used_edges
                candidates = []
                for h in range(20):
                    if h in used_hosts:
                        continue
                    edge = (min(hp, h), max(hp, h))
                    if edge in used_edges:
                        continue
                    candidates.append(h)
                
                if not candidates:
                    return False
                
                # Heuristic: Prefer candidates that have more unused neighbors
                # for future children (if any)
                if children20[v]:
                    def score_candidate(h):
                        count = 0
                        for nh in range(20):
                            if nh in used_hosts:
                                continue
                            e = (min(h, nh), max(h, nh))
                            if e not in used_edges:
                                count += 1
                        return count
                    candidates.sort(key=score_candidate, reverse=True)
                else:
                    rng.shuffle(candidates)
                
                for h in candidates:
                    host20[v] = h
                    used_hosts.add(h)
                    
                    if backtrack_t20(idx + 1):
                        return True
                    
                    host20[v] = None
                    used_hosts.remove(h)
                
                return False
            
            # Start backtracking for T20
            if backtrack_t20(1):
                # T20 placed successfully
                embeddings[19] = host20[:]
                for u, v in t20_edges:
                    used_edges.add((min(host20[u], host20[v]), max(host20[u], host20[v])))
                
                # Now place T19, T18, ..., T2
                success = True
                for k in range(19, 1, -1):
                    if time.time() > deadline - 1:
                        success = False
                        break
                    
                    t_edges = trees[k-1]
                    t_adj = [[] for _ in range(k)]
                    for u, v in t_edges:
                        t_adj[u].append(v)
                        t_adj[v].append(u)
                        
                    # Root at 0
                    par = [-1] * k
                    ord_list = [0]
                    vis = [False] * k
                    vis[0] = True
                    q = [0]
                    while q:
                        u = q.pop(0)
                        for v in t_adj[u]:
                            if not vis[v]:
                                vis[v] = True
                                par[v] = u
                                q.append(v)
                                ord_list.append(v)
                    
                    # Precompute children
                    children_k = [[] for _ in range(k)]
                    for v in range(1, k):
                        children_k[par[v]].append(v)
                    
                    host_k = [None] * k
                    used_hosts_k = set()
                    
                    def backtrack_k(idx):
                        if time.time() > deadline - 1:
                            return False
                        if idx == k:
                            return True
                        
                        v = ord_list[idx]
                        p = par[v]
                        hp = host_k[p] if p != -1 else None
                        
                        candidates = []
                        for h in range(20):
                            if h in used_hosts_k:
                                continue
                            if hp is not None:
                                edge = (min(hp, h), max(hp, h))
                                if edge in used_edges:
                                    continue
                            candidates.append(h)
                        
                        if not candidates:
                            return False
                        
                        # Heuristic: Prefer candidates that have more unused neighbors
                        if children_k[v]:
                            def score_candidate_k(h):
                                count = 0
                                for nh in range(20):
                                    if nh in used_hosts_k:
                                        continue
                                    e = (min(h, nh), max(h, nh))
                                    if e not in used_edges:
                                        count += 1
                                return count
                            candidates.sort(key=score_candidate_k, reverse=True)
                        else:
                            rng.shuffle(candidates)
                        
                        for h in candidates:
                            host_k[v] = h
                            used_hosts_k.add(h)
                            
                            if backtrack_k(idx + 1):
                                return True
                            
                            host_k[v] = None
                            used_hosts_k.remove(h)
                        
                        return False
                    
                    if backtrack_k(0):
                        embeddings[k-1] = host_k[:]
                        for u, v in t_edges:
                            used_edges.add((min(host_k[u], host_k[v]), max(host_k[u], host_k[v])))
                    else:
                        success = False
                        break
                
                if success:
                    embeddings[0] = [0]
                    score = 0
                    for k in range(1, 21):
                        if embeddings[k-1] is not None:
                            score += k - 1
                    if score > best_score:
                        best_score = score
                        best_embeddings = [e[:] if e is not None else None for e in embeddings]
                    
                    if best_score == 190:
                        return best_embeddings
                
            # If T20 failed, try again with a different randomization

        return best_embeddings

    result = solve_packing(trees, deadline)
    
    # If result is None or incomplete, try to salvage
    if result is None:
        result = [None] * 20
        result[0] = [0]
        
    # Ensure types are correct
    for i in range(20):
        if result[i] is not None:
            if len(result[i]) != i + 1:
                result[i] = None
            else:
                # Convert to list of ints
                result[i] = [int(x) for x in result[i]]
                if len(set(result[i])) != i + 1:
                    result[i] = None

    return result
