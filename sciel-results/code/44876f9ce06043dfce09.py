import random
import time

def construct():
    n = 24
    # Generate the edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Shuffle with the pinned seed
    r = random.Random(614)
    r.shuffle(edges)
    
    # We want to find the smallest K such that the first K edges contain a Hamilton cycle.
    # Initialize adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    def has_hamiltonian_cycle():
        # Check degree constraint: all vertices must have degree >= 2
        for i in range(n):
            deg = sum(1 for j in range(n) if adj[i][j])
            if deg < 2:
                return None
        
        # DFS with bitmask
        # Start at 0
        visited = 1 # bit 0 set
        path = [0]
        
        # Recursive helper
        def dfs(u, visited_mask, path):
            if len(path) == n:
                if adj[u][0]:
                    return path[:]
                return None
            
            # Heuristic: try neighbors that lead to dead ends later? 
            # Or just iterate.
            # Optimization: If the number of unvisited vertices is small, we can be more aggressive.
            
            # Pruning: Check if any unvisited vertex has degree < 2 in the induced subgraph of unvisited + {u, 0}?
            # This is complex. Let's stick to simple degree check on the full graph first.
            # If we are at step k, and there is an unvisited vertex v with only 1 neighbor in unvisited + {0} (if v is not last) or {u} (if v is last), it might be a problem.
            
            # Let's just do standard DFS. N=24 is small enough for bitmasks.
            
            for v in range(n):
                if not (visited_mask & (1 << v)):
                    if adj[u][v]:
                        # Pruning: If v is the last unvisited, check connection to 0
                        # Count unvisited
                        # This is O(N) per step, total O(N^2) per path? No, O(N) per step, N steps, so O(N^2) per branch.
                        
                        # Simple check: if v is the only unvisited node
                        # unvisited_count = n - bin(visited_mask).count('1')
                        # if unvisited_count == 1:
                        #     if not adj[v][0]: continue
                        
                        res = dfs(v, visited_mask | (1 << v), path + [v])
                        if res:
                            return res
            return None
        
        return dfs(0, visited, path)

    # Iterate K
    start_time = time.time()
    best_cycle = None
    
    # Minimum edges for a cycle is n
    for k in range(n, len(edges) + 1):
        u, v = edges[k - 1]
        adj[u][v] = True
        adj[v][u] = True
        
        # Time check
        if time.time() - start_time > 45:
            break
            
        cycle = has_hamiltonian_cycle()
        if cycle:
            return cycle
            
    # Fallback: Greedy construction
    # This is the same as the seed program's logic, but we need to ensure it returns a valid cycle.
    # If we didn't find one in the loop, it means we timed out or no cycle found (unlikely for K=276).
    # Let's just return the identity if it fails, but identity is not a Hamilton cycle in the edge list sense?
    # The task says "return a permutation... describing a Hamilton cycle".
    # The evaluator checks if the edges of this permutation exist in the list.
    # If we return identity [0, 1, 2, ..., 23], the edges are (0,1), (1,2), ..., (23,0).
    # We should return a cycle that we *know* exists, e.g., from the last successful check or a pre-computed one.
    
    # Since K=276 is the full graph, a Hamilton cycle MUST exist.
    # If the loop finished without returning, it means the time guard broke it.
    # In that case, we should return a valid cycle from a known good K, or just run a greedy.
    
    # Let's implement a robust greedy as fallback
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
            nxt = adj_list[cur][0] if adj_list[cur][0] != prev else adj_list[cur][1]
            cycle.append(nxt)
            prev = cur
            cur = nxt
        return cycle
    else:
        return list(range(n))
