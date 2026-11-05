import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 16
    # Start with K8,8 (bipartite)
    # Vertices 0-7 in L, 8-15 in R
    adj = [[False] * n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = True
            adj[v][u] = True
            
    def find_q3(adj_matrix):
        # Returns list of 8 vertices forming a Q3 subgraph, or None
        L = list(range(8))
        R = list(range(8, 16))
        
        for l_subset in itertools.combinations(L, 4):
            for r_subset in itertools.combinations(R, 4):
                subset = list(l_subset) + list(r_subset)
                
                # Collect missing edges in the 4x4 grid
                missing = []
                for i in range(4):
                    for j in range(4):
                        if not adj_matrix[subset[i]][subset[4+j]]:
                            missing.append((i, j))
                            
                # Check if missing edges form a matching
                used_l = set()
                used_r = set()
                is_matching = True
                for i, j in missing:
                    if i in used_l or j in used_r:
                        is_matching = False
                        break
                    used_l.add(i)
                    used_r.add(j)
                    
                if is_matching:
                    return subset
                    
        return None

    # Deletion Loop
    while time.time() < deadline - 5:
        q3_vertices = find_q3(adj)
        if q3_vertices is None:
            break
            
        # Remove an edge from this Q3
        removed = False
        for i in range(4):
            for j in range(4):
                u = q3_vertices[i]
                v = q3_vertices[4+j]
                if adj[u][v]:
                    adj[u][v] = False
                    adj[v][u] = False
                    removed = True
                    break
            if removed:
                break
                
        if not removed:
            break
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges
