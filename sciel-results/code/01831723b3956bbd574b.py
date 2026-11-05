import random
import time

def construct():
    n = 24
    start_time = time.time()
    
    # Generate the edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Shuffle with the pinned seed
    r = random.Random(614)
    r.shuffle(edges)
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # We want to find a Hamilton cycle with minimum max edge index.
    # Let's use a heuristic search.
    # Since n=24 is small, we can try to construct a cycle using edges with low indices.
    
    # Strategy:
    # 1. Sort edges by index (they already are).
    # 2. Try to build a cycle using only edges with index < K.
    # 3. Increase K until a cycle is found.
    # 4. To speed up, we can use a greedy/DFS approach with pruning.
    
    # However, checking every K from 24 to 276 might be slow if we do full DFS each time.
    # Instead, let's try a randomized local search or a more efficient exact method.
    
    # Let's try a backtracking search with strong pruning for a given K.
    # We'll iterate K from n upwards.
    
    adj = [set() for _ in range(n)]
    
    def find_cycle(K):
        # Reset adj
        for i in range(n):
            adj[i].clear()
        for i in range(K):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) < 2:
                return None
        
        # DFS
        # Start at 0
        path = [0]
        visited = {0}
        
        def dfs(u):
            if len(path) == n:
                if 0 in adj[u]:
                    return path[:]
                return None
            
            # Heuristic: order neighbors by degree in the remaining graph?
            # Or just iterate.
            # Pruning: If any unvisited vertex has degree 0 or 1 in the induced subgraph of unvisited + {u},
            # and it's not the last one, we might be stuck.
            # This is expensive. Let's try a simpler version first.
            
            # Sort neighbors to try promising ones first?
            # Maybe try neighbors with lower degree first (Warnsdorff's rule heuristic for Hamiltonian paths)?
            neighbors = list(adj[u])
            # Filter visited
            unvisited_neighbors = [v for v in neighbors if v not in visited]
            
            # If we are at the last step (len(path) == n-1), the only neighbor must be 0
            if len(path) == n - 1:
                if 0 in unvisited_neighbors:
                    return path + [0]
                return None
            
            # Heuristic: sort by degree of unvisited neighbors?
            # Let's just try all.
            for v in unvisited_neighbors:
                # Pruning: If v is the only unvisited node, check connection to 0
                # Count unvisited
                unvisited_count = n - len(path)
                if unvisited_count == 1:
                    if 0 not in adj[v]:
                        continue
                
                visited.add(v)
                path.append(v)
                
                res = dfs(v)
                if res:
                    return res
                
                visited.remove(v)
                path.pop()
            
            return None
        
        return dfs(0)
    
    # Iterate K
    # Minimum K is n (24).
    # We stop when time is running out.
    deadline = start_time + 45
    
    best_cycle = None
    
    for k in range(n, len(edges) + 1):
        if time.time() > deadline:
            break
        
        cycle = find_cycle(k)
        if cycle:
            best_cycle = cycle
            break
    
    if best_cycle:
        return best_cycle
    
    # Fallback: Greedy construction (like the seed)
    # This ensures we return *something* valid if the exact search fails or times out.
    
    parent = list(range(n))
    rank = [0] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px == py: return False
        if rank[px] < rank[py]: px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]: rank[px] += 1
        return True

    degree = [0] * n
    comp_size = [1] * n
    selected_edges = []
    
    for i, (u, v) in enumerate(edges):
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        fu, fv = find(u), find(v)
        if fu == fv:
            if comp_size[fu] == n:
                degree[u] += 1
                degree[v] += 1
                selected_edges.append((u, v))
                break
            else:
                continue
        else:
            su, sv = comp_size[fu], comp_size[fv]
            degree[u] += 1
            degree[v] += 1
            selected_edges.append((u, v))
            new_size = su + sv
            if su < sv:
                parent[fu] = fv
                comp_size[fv] = new_size
            else:
                parent[fv] = fu
                comp_size[fu] = new_size
                
    if len(selected_edges) == n:
        adj_list = {i: [] for i in range(n)}
        for u, v in selected_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        cycle = [0]
        prev = -1
        cur = 0
        for _ in range(n - 1):
            # Pick next
            candidates = [x for x in adj_list[cur] if x != prev]
            if not candidates:
                candidates = adj_list[cur]
            nxt = candidates[0]
            cycle.append(nxt)
            prev = cur
            cur = nxt
        return cycle
    else:
        return list(range(n))
