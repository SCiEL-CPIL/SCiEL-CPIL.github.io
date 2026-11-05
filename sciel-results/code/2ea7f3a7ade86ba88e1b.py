import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    def get_degree_sum_score(adj):
        """Calculate the maximum degree sum over all triangles."""
        degs = adj.sum(axis=1)
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # Common neighbors
                    common = np.where(adj[u] & adj[v] > 0)[0]
                    for w in common:
                        if w > v:
                            s = degs[u] + degs[v] + degs[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    rng = np.random.default_rng(42)
    
    def random_graph_with_k_edges(k):
        edges = set()
        while len(edges) < k:
            u, v = rng.integers(0, n, 2)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        adj_mat = np.zeros((n, n), dtype=int)
        for u, v in edges:
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
        return adj_mat

    # Initialize with multiple random graphs to find a good starting point
    best_adj = None
    best_score = float('inf')
    
    # Try K_8,8 + 1 edge
    adj0 = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            adj0[u, v] = 1
            adj0[v, u] = 1
    adj0[0, 1] = 1
    adj0[1, 0] = 1
    
    candidates = [adj0]
    for _ in range(5):
        candidates.append(random_graph_with_k_edges(target_edges))
        
    for adj in candidates:
        score = get_degree_sum_score(adj)
        if score < best_score:
            best_score = score
            best_adj = adj.copy()

    current_adj = best_adj.copy()
    current_score = best_score
    
    # Maintain lists of edges and non-edges for O(1) sampling
    edge_list = [(u, v) for u in range(n) for v in range(u+1, n) if current_adj[u, v]]
    non_edge_list = [(u, v) for u in range(n) for v in range(u+1, n) if not current_adj[u, v]]
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    iterations = 500000
    
    for i in range(iterations):
        if not edge_list or not non_edge_list:
            break
            
        e_idx = rng.integers(0, len(edge_list))
        u, v = edge_list[e_idx]
        
        ne_idx = rng.integers(0, len(non_edge_list))
        x, y = non_edge_list[ne_idx]
        
        if (u == x and v == y) or (u == y and v == x):
            continue

        # Apply move
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[x, y] = 1
        current_adj[y, x] = 1
        
        new_score = get_degree_sum_score(current_adj)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            # Swap in lists
            edge_list[e_idx] = (x, y)
            non_edge_list[ne_idx] = (u, v)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
        else:
            # Revert
            current_adj[u, v] = 1
            current_adj[v, u] = 1
            current_adj[x, y] = 0
            current_adj[y, x] = 0
            
        T *= cooling
        if T < T_min:
            T = T_min

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
