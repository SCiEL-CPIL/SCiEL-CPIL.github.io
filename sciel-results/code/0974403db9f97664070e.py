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
    deadline = start_time + 110  # 10s safety margin
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute tree properties
    tree_info = []
    for i in range(n):
        k = i + 1
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'adj': [[]], 'deg': [0]})
            continue
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for (a, b) in trees[i]:
            adj[a].append(b)
            adj[b].append(a)
            deg[a] += 1
            deg[b] += 1
        tree_info.append({'k': k, 'edges': trees[i], 'adj': adj, 'deg': deg})
        
    # Order: largest first
    order = list(range(n - 1, -1, -1))
    
    used_edges = [[False] * n for _ in range(n)]
    used_hosts = [False] * n
    result = [None] * n
    
    # Helper to check if edge is free
    def edge_free(u, v):
        if u == v: return False
        return not used_edges[u][v]
        
    def mark_edge(u, v, val):
        used_edges[u][v] = val
        used_edges[v][u] = val
        
    def embed_tree(tree_idx, max_time_per_tree=5.0):
        """
        Attempts to embed tree tree_idx.
        Returns list of host vertices or None.
        """
        t_start = time.time()
        info = tree_info[tree_idx]
        k = info['k']
        
        if k == 1:
            # Find any free host
            for h in range(n):
                if not used_hosts[h]:
                    used_hosts[h] = True
                    return [h]
            return None
            
        if k == 2:
            # Find any free edge
            for i in range(n):
                if used_hosts[i]: continue
                for j in range(i+1, n):
                    if used_hosts[j]: continue
                    if edge_free(i, j):
                        used_hosts[i] = True
                        used_hosts[j] = True
                        mark_edge(i, j, True)
                        return [i, j]
            return None
            
        adj = info['adj']
        deg = info['deg']
        
        # Try different roots? Start with highest degree.
        # Sort vertices by degree descending for root selection candidates
        root_candidates = sorted(range(k), key=lambda v: -deg[v])
        
        for root in root_candidates:
            # BFS order from root
            parent = [-1] * k
            visited = [False] * k
            visited[root] = True
            queue = [root]
            assign_order = [root]
            while queue:
                u = queue.pop(0)
                for w in adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        parent[w] = u
                        assign_order.append(w)
                        queue.append(w)
                        
            # Check if time is up
            if time.time() - t_start > max_time_per_tree:
                break
                
            host_map = [-1] * k
            local_used_hosts = set()
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                if pos == k:
                    return True
                    
                v = assign_order[pos]
                
                if pos == 0:
                    # Root
                    # Symmetry breaking: if this is the first tree embedded, fix to 0
                    # We don't easily know if it's the first, but we can check result
                    # Let's just iterate all free hosts
                    candidates = [h for h in range(n) if not used_hosts[h]]
                else:
                    pv = parent[v]
                    ph = host_map[pv]
                    candidates = []
                    for h in range(n):
                        if used_hosts[h] or h in local_used_hosts:
                            continue
                        if edge_free(ph, h):
                            candidates.append(h)
                            
                # Heuristic: Sort candidates by degree in remaining graph?
                # Or just randomize?
                # Let's sort by how many free edges they have (descending)
                # This is expensive to compute for all candidates every time.
                # Simple heuristic: try hosts with higher index first? No.
                # Let's just iterate.
                
                for h in candidates:
                    host_map[pos] = h
                    local_used_hosts.add(h)
                    
                    if pos > 0:
                        pv = parent[v]
                        ph = host_map[pv]
                        mark_edge(ph, h, True)
                    
                    if backtrack(pos + 1):
                        return True
                        
                    if pos > 0:
                        pv = parent[v]
                        ph = host_map[pv]
                        mark_edge(ph, h, False)
                        
                    local_used_hosts.discard(h)
                    host_map[pos] = -1
                    
                return False
                
            if backtrack(0):
                # Commit
                for pos, v in enumerate(assign_order):
                    h = host_map[v]
                    used_hosts[h] = True
                # Edges are already marked in backtrack success path?
                # Wait, backtrack marks edges on the way down, but doesn't unmark on success.
                # So they are marked.
                return [host_map[v] for v in range(k)]
                
        return None

    for tree_idx in order:
        if time.time() > deadline:
            break
            
        res = embed_tree(tree_idx)
        if res is not None:
            result[tree_idx] = res
        else:
            result[tree_idx] = None
            
    return result
