"""Spectral optimization for minimizing the largest book in a 40-vertex, 401-edge graph."""
import numpy as np
import random

def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book
    (max over edges of number of triangles containing that edge).
    
    Approach: Spectral optimization via local search.
    1. Initialize with a quasi-random regular-like graph
    2. Use local edge swaps to flatten the triangle distribution (A^2 on edges)
    3. Guide swaps to minimize max A^2[u][v] over edges (u,v)
    """
    n = 40
    target_edges = 401
    
    def compute_books(adj):
        """For each edge (u,v), book size = number of common neighbors = (A^2)[u][v]"""
        A2 = adj @ adj
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    books[(u, v)] = A2[u, v]
        return books
    
    def max_book(adj):
        """Compute the maximum book size over all edges"""
        A2 = adj @ adj
        max_b = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    if A2[u, v] > max_b:
                        max_b = A2[u, v]
        return max_b
    
    def get_high_book_edges(adj, threshold):
        """Get edges with book size >= threshold"""
        A2 = adj @ adj
        high = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] and A2[u, v] >= threshold:
                    high.append((u, v, A2[u, v]))
        return high
    
    def get_non_edges(adj):
        """Get all non-edges"""
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]:
                    non_edges.append((u, v))
        return non_edges
    
    def get_edges(adj):
        """Get all edges"""
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    edges.append((u, v))
        return edges
    
    # Step 1: Initialize with a quasi-random regular-like graph
    # Start with a random graph with 401 edges
    random.seed(42)
    
    # Create a random initial graph
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_possible)
    
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in all_possible[:target_edges]:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Initial max book
    current_max = max_book(adj)
    best_adj = adj.copy()
    best_max = current_max
    
    # Step 2: Local search - try edge swaps to reduce max book
    max_iterations = 8000
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        # Get edges with high book sizes
        high_edges = get_high_book_edges(adj, current_max)
        
        if not high_edges:
            break
        
        if current_max <= 6:
            break
        
        improved = False
        random.shuffle(high_edges)
        
        # Try to reduce the book size of high-book edges
        for (u, v, book_size) in high_edges[:8]:
            if book_size < current_max:
                continue
            
            # Try removing (u,v) and adding a different edge
            # Temporarily remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            
            A2_temp = adj @ adj
            
            # Find candidate non-edges to add
            candidates = []
            for w in range(n):
                for x in range(w+1, n):
                    if adj[w, x]:
                        continue
                    new_book_wx = A2_temp[w, x]
                    candidates.append((w, x, new_book_wx))
            
            # Sort by new book size (ascending) and try a few
            candidates.sort(key=lambda c: c[2])
            
            best_new_max = current_max + 100
            best_new_edge = None
            
            for w, x, new_book in candidates[:30]:
                # Add edge (w,x) temporarily
                adj[w, x] = 1
                adj[x, w] = 1
                
                # Compute new max book
                A2_check = adj @ adj
                max_b = 0
                for a in range(n):
                    for b in range(a+1, n):
                        if adj[a, b]:
                            if A2_check[a, b] > max_b:
                                max_b = A2_check[a, b]
                
                # Remove (w,x)
                adj[w, x] = 0
                adj[x, w] = 0
                
                if max_b < best_new_max:
                    best_new_max = max_b
                    best_new_edge = (w, x)
            
            # Restore (u,v)
            adj[u, v] = 1
            adj[v, u] = 1
            
            # Apply best swap if it improves
            if best_new_edge is not None and best_new_max < current_max:
                adj[u, v] = 0
                adj[v, u] = 0
                w, x = best_new_edge
                adj[w, x] = 1
                adj[x, w] = 1
                current_max = best_new_max
                improved = True
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_adj = adj.copy()
                break
        
        if not improved:
            no_improve_count += 1
            # If stuck, try a random perturbation
            if no_improve_count > 30:
                edge_list = get_edges(adj)
                non_edge_list = get_non_edges(adj)
                
                if edge_list and non_edge_list:
                    u, v = random.choice(edge_list)
                    w, x = random.choice(non_edge_list)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[w, x] = 1
                    adj[x, w] = 1
                    current_max = max_book(adj)
                    no_improve_count = 0
                    if current_max < best_max:
                        best_max = current_max
                        best_adj = adj.copy()
                else:
                    break
    
    # Use the best found
    adj = best_adj
    
    # Extract edges from adjacency matrix
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                result.append([u, v])
    
    # Ensure exactly 401 edges
    if len(result) != target_edges:
        # Fallback to a valid construction
        result = []
        for a in range(20):
            for b in range(20, 40):
                result.append([a, b])
        result.append([0, 1])
    
    return result
