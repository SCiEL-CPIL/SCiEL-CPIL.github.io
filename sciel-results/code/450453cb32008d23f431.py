import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 16
    vertices = list(range(n))
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute all 8-subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    def find_q3(adj):
        # adj is list of 16 ints (bitmasks)
        for subset in subsets_8:
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check if induced subgraph has 12 edges (sum of degrees = 24)
            deg_sum = 0
            for u in subset:
                deg_sum += bin(adj[u] & mask).count("1")
            
            if deg_sum != 24:
                continue
                
            # Check degrees are all 3
            valid_degrees = True
            for u in subset:
                if bin(adj[u] & mask).count("1") != 3:
                    valid_degrees = False
                    break
            
            if not valid_degrees:
                continue
                
            # Check bipartite
            visited = [0] * n
            color = [0] * n
            start_node = subset[0]
            visited[start_node] = 1
            color[start_node] = 0
            queue = [start_node]
            is_bipartite = True
            
            while queue:
                u = queue.pop(0)
                neighbors_mask = adj[u] & mask
                for v in subset:
                    if u == v: continue
                    if (neighbors_mask >> v) & 1:
                        if visited[v] == 0:
                            visited[v] = 1
                            color[v] = 1 - color[u]
                            queue.append(v)
                        else:
                            if color[v] == color[u]:
                                is_bipartite = False
                                break
                if not is_bipartite:
                    break
            
            if is_bipartite:
                return subset
                
        return None

    best_edges_count = 0
    best_adj = [0] * n
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random graph with ~80% density
        adj = [0] * n
        for u, v in all_edges:
            if random.random() < 0.8:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Repair Phase
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            q3_subset = find_q3(adj)
            if not q3_subset:
                break
            
            # Remove a random edge from this Q3
            q3_edges = []
            for i, u in enumerate(q3_subset):
                for v in q3_subset[i+1:]:
                    if (adj[u] >> v) & 1:
                        q3_edges.append((u, v))
            
            if q3_edges:
                u, v = random.choice(q3_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Local Search Phase
        missing_edges = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
        random.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in missing_edges:
                if (adj[u] >> v) & 1:
                    continue
                
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if find_q3(adj) is None:
                    # Keep it
                    improved = True
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # Count edges
        count = sum(bin(adj[i]).count("1") for i in range(n)) // 2
        
        if count > best_edges_count:
            best_edges_count = count
            best_adj = adj[:]

    # Convert best_adj to list of edges
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                final_edges.append([u, v])
                
    return final_edges
