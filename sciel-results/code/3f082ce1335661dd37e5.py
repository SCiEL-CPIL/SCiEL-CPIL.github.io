import random
import time

def construct():
    n = 40
    start_time = time.time()
    time_limit = 115.0
    
    def get_neighbors(adj, u):
        return [i for i in range(n) if adj[u][i]]

    def has_path_len_5(adj, u, v):
        """
        Checks if there is a path of length 5 from u to v in adj.
        Path: u-a-b-c-d-v
        """
        nu = get_neighbors(adj, u)
        nv = get_neighbors(adj, v)
        
        for a in nu:
            if a == v:
                continue
            na = get_neighbors(adj, a)
            for b in na:
                if b == u or b == v or b == a:
                    continue
                nb = get_neighbors(adj, b)
                for c in nb:
                    if c == u or c == v or c == a or c == b:
                        continue
                    nc = get_neighbors(adj, c)
                    for d in nc:
                        if d == u or d == v or d == a or d == b or d == c:
                            continue
                        if adj[d][v]:
                            return True
        return False

    def find_c6_edge(adj):
        """
        Finds an edge that is part of a C6, or returns None if graph is C6-free.
        """
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        
        if not non_edges:
            return None
            
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if has_path_len_5(adj, u, v):
                # Re-run to find the specific edge
                nu = get_neighbors(adj, u)
                for a in nu:
                    if a == v: continue
                    na = get_neighbors(adj, a)
                    for b in na:
                        if b == u or b == v or b == a: continue
                        nb = get_neighbors(adj, b)
                        for c in nb:
                            if c == u or c == v or c == a or c == b: continue
                            nc = get_neighbors(adj, c)
                            for d in nc:
                                if d == u or d == v or d == a or d == b or d == c: continue
                                if adj[d][v]:
                                    return (u, a)
        return None

    best_adj = None
    best_count = 0
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(restart * 42 + 1)
        adj = [bytearray(n) for _ in range(n)]
        
        p = 0.2
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Phase 1: Prune C6s
        while True:
            if time.time() - start_time > time_limit:
                break
            edge_to_remove = find_c6_edge(adj)
            if edge_to_remove is None:
                break
            u, v = edge_to_remove
            adj[u][v] = 0
            adj[v][u] = 0
            
        if time.time() - start_time > time_limit:
            break
            
        current_count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        
        if current_count > best_count:
            best_count = current_count
            best_adj = [row[:] for row in adj]
            
        # Phase 2: Add edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() - start_time > time_limit:
                break
            if not has_path_len_5(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                current_count += 1
                
        if current_count > best_count:
            best_count = current_count
            best_adj = [row[:] for row in adj]

    if best_adj is None:
        return []
        
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                result.append([i, j])
    return result
