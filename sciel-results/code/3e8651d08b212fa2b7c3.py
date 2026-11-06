"""Spectral optimization approach for minimizing max triangle degree sum."""
import numpy as np
import random

def construct():
    n = 16
    target_edges = 65
    
    def spectral_radius_power(adj, max_iter=200, tol=1e-8):
        """
        Estimate the largest eigenvalue of the adjacency matrix using power iteration.
        adj: numpy array of shape (n, n) with 0/1 values.
        """
        n = adj.shape[0]
        # Start with a random vector to avoid symmetry issues
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        for _ in range(max_iter):
            w = adj @ v
            norm = np.linalg.norm(w)
            if norm < 1e-10:
                return 0.0
            w = w / norm
            # Check convergence
            if np.linalg.norm(w - v) < tol:
                break
            v = w
        # Rayleigh quotient for better estimate
        return np.dot(v, adj @ v)
        
    def compute_actual_score(adj):
        """Compute the maximum degree sum over all triangles."""
        deg = adj.sum(axis=1)
        max_sum = 0
        # Find all triangles
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    common_mask = adj[i] & adj[j]
                    common_indices = np.where(common_mask)[0]
                    for k in common_indices:
                        if k > j:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def local_search_spectral(adj, max_iter=200):
        """
        Perform local search to minimize spectral radius.
        adj: numpy array of shape (n, n) with 0/1 values.
        Returns the improved adj matrix.
        """
        current_lambda = spectral_radius_power(adj)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # List of edges
            edges_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges_list.append((i, j))
                        
            # List of non-edges
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i, j]:
                        non_edges.append((i, j))
                        
            # Try all swaps
            best_delta = 0
            best_move = None
            
            # Optimization: Precompute degrees? No, lambda changes.
            # We just evaluate lambda for each candidate swap.
            
            for u, v in edges_list:
                for x, y in non_edges:
                    # Perform swap
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[x, y] = 1
                    adj[y, x] = 1
                    
                    new_lambda = spectral_radius_power(adj)
                    
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
    num_restarts = 50
    random.seed(42)
    np.random.seed(42)
    
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
        elif r == 1:
            # K_8,8 + (0,8) (cross edge? No, 0 is in A, 8 is in B, so this is already an edge in K8,8)
            # Let's add an edge within B
            for i in range(8):
                for j in range(8, 16):
                    adj[i, j] = 1
                    adj[j, i] = 1
            adj[8, 9] = 1
            adj[9, 8] = 1
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
            # This helps escape the immediate local minimum of K8,8+1
            for _ in range(20):
                edges_list = []
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i, j]:
                            edges_list.append((i, j))
                        else:
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
        adj_improved = local_search_spectral(adj, max_iter=100)
        
        # Evaluate actual score
        score = compute_actual_score(adj_improved)
        
        if score < best_score:
            best_score = score
            best_adj = adj_improved.copy()
            
        # Early stopping if we hit the theoretical lower bound?
        # Conjectured asymptotic is ~1.464 * 16 = 23.42.
        # Integer score, so 24 is the target. 23 is possible?
        # If we find 24, it's very good. If we find 23, it's better.
        if best_score <= 23:
            break

    # Final check: ensure we return a valid graph
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
