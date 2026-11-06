"""Spectral optimization approach for minimizing max triangle degree sum."""
import numpy as np
import random

def construct():
    n = 16
    target_edges = 65
    
    def compute_spectral_radius(adj):
        """Compute the largest eigenvalue of the adjacency matrix."""
        # For small N, eigvalsh is robust and fast
        eigvals = np.linalg.eigvalsh(adj)
        return eigvals[-1]
        
    def compute_actual_score(adj):
        """Compute the maximum degree sum over all triangles."""
        deg = adj.sum(axis=1)
        max_sum = 0
        # Find all triangles
        # Use bitmasks or loops. For N=16, loops are fine.
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    # Common neighbors
                    # adj[i] & adj[j] gives a vector of common neighbors
                    common_mask = adj[i] & adj[j]
                    # Indices of common neighbors
                    common_indices = np.where(common_mask)[0]
                    for k in common_indices:
                        if k > j: # Count each triangle once
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def local_search_spectral(adj, max_iter=100):
        """
        Perform local search to minimize spectral radius.
        adj: numpy array of shape (n, n) with 0/1 values.
        Returns the improved adj matrix.
        """
        current_lambda = compute_spectral_radius(adj)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # List of edges
            edges = np.where(adj > 0)
            edges_list = [(edges[0][i], edges[1][i]) for i in range(len(edges[0])) if edges[0][i] < edges[1][i]]
            
            # List of non-edges
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] == 0:
                        non_edges.append((i, j))
                        
            # Try all swaps
            best_delta = 0
            best_move = None
            
            for u, v in edges_list:
                for x, y in non_edges:
                    # Skip if swapping an edge with itself (not possible as one is edge, one is non-edge)
                    # Perform swap
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[x, y] = 1
                    adj[y, x] = 1
                    
                    new_lambda = compute_spectral_radius(adj)
                    
                    # Revert
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[x, y] = 0
                    adj[y, x] = 0
                    
                    delta = new_lambda - current_lambda
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (u, v, x, y)
                        
            if best_move is not None:
                u, v, x, y = best_move
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1
                current_lambda += best_delta
                improved = True
                
        return adj

    best_score = float('inf')
    best_adj = None
    
    # Try multiple initial configurations
    # 1. K_8,8 + 1 edge
    # 2. Random graphs with 65 edges
    
    num_restarts = 20
    random.seed(42)
    
    for r in range(num_restarts):
        adj = np.zeros((n, n), dtype=np.int8)
        
        if r == 0:
            # K_8,8 + (0,1)
            for i in range(8):
                for j in range(8, 16):
                    adj[i, j] = 1
                    adj[j, i] = 1
            adj[0, 1] = 1
            adj[1, 0] = 1
        else:
            # Random graph with 65 edges
            # Start with K_8,8 (64 edges) and add 1 random intra-part edge
            for i in range(8):
                for j in range(8, 16):
                    adj[i, j] = 1
                    adj[j, i] = 1
            
            # Add one random edge within part A or B
            if r % 2 == 0:
                u, v = random.sample(range(8), 2)
            else:
                u, v = random.sample(range(8, 16), 2)
            adj[u, v] = 1
            adj[v, u] = 1
            
            # Optionally, perform some random swaps to diversify
            for _ in range(10):
                edges = np.where(adj > 0)
                edges_list = [(edges[0][i], edges[1][i]) for i in range(len(edges[0])) if edges[0][i] < edges[1][i]]
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i, j] == 0:
                            non_edges.append((i, j))
                
                if not edges_list or not non_edges:
                    continue
                    
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges)
                
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1

        # Run spectral local search
        adj_improved = local_search_spectral(adj, max_iter=50)
        
        # Evaluate actual score
        score = compute_actual_score(adj_improved)
        
        if score < best_score:
            best_score = score
            best_adj = adj_improved.copy()
            
        # If we found a very good score, we might stop early?
        # The conjectured value is ~23.4. If we find 24 or lower, that's good.
        if best_score <= 24:
            # Continue to see if we can go lower, but this is a strong candidate
            pass

    # Final check: ensure we return a valid graph
    # If best_adj is None, fall back to K_8,8 + (0,1)
    if best_adj is None:
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(8):
            for j in range(8, 16):
                adj[i, j] = 1
                adj[j, i] = 1
        adj[0, 1] = 1
        adj[1, 0] = 1
        best_adj = adj

    # Convert to edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
