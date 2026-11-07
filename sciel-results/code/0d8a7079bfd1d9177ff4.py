import time
import random
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 140  # 10s safety margin
    
    trees = instance_trees(20, 611)
    
    # used_edges stores edges of K20 that are already occupied
    used_edges = set()
    result = [None] * 20
    
    # Process trees from largest to smallest to maximize edge count
    for k in range(20, 0, -1):
        if time.time() > deadline:
            break
            
        if k == 1:
            # T1 is a single vertex. Any host works.
            # We pick an arbitrary host, say 0.
            # But wait, if 0 is used by another tree's vertex, is that a problem?
            # "Embedded trees must be edge-disjoint". Vertices can overlap.
            # So T1 can map to any vertex. Let's pick 0.
            result[k-1] = [0]
            continue
            
        edges = trees[k-1]
        
        # Build tree structure for T_k
        # We need to root it and get BFS order
        # Since it's a tree, we can build adjacency list
        adj = {i: [] for i in range(k)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Root at 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    order.append(v)
                    
        # order is now BFS order
        # We will use backtracking to find an embedding
        
        def solve_tree(order, parent, k, used_edges):
            mapping = {}
            assigned_hosts = set()
            
            def dfs(idx):
                if idx == k:
                    return True
                    
                v = order[idx]
                
                if idx == 0:
                    # Root: any host
                    candidates = list(range(20))
                else:
                    p = parent[v]
                    h_p = mapping[p]
                    candidates = []
                    for u in range(20):
                        if u in assigned_hosts:
                            continue
                        if u == h_p:
                            continue
                        e = (min(h_p, u), max(h_p, u))
                        if e in used_edges:
                            continue
                        candidates.append(u)
                
                # Heuristic: sort candidates by degree in remaining graph?
                # For K20 minus used edges, degree is 19 - (number of used edges incident to u).
                # We prefer hosts with more available edges to keep options open.
                
                def heuristic_score(u):
                    deg = 0
                    for w in range(20):
                        if w == u:
                            continue
                        e = (min(u, w), max(u, w))
                        if e not in used_edges:
                            deg += 1
                    return deg
                
                candidates.sort(key=heuristic_score, reverse=True)
                
                for u in candidates:
                    mapping[v] = u
                    assigned_hosts.add(u)
                    
                    if dfs(idx + 1):
                        return True
                        
                    del mapping[v]
                    assigned_hosts.remove(u)
                    
                return False
                
            if dfs(0):
                return mapping
            return None

        mapping = solve_tree(order, parent, k, used_edges)
        
        if mapping is not None:
            # Update used_edges
            for u, v in edges:
                h_u = mapping[u]
                h_v = mapping[v]
                e = (min(h_u, h_v), max(h_u, h_v))
                used_edges.add(e)
            
            # Create the result list
            # Position i is the image of vertex i
            res_list = [mapping[i] for i in range(k)]
            result[k-1] = res_list
            
    return result
