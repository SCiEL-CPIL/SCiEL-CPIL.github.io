"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book.
    Approach: Spectral optimization via local search.
    1. Initialize with a random graph with 401 edges.
    2. Use local edge swaps to minimize the maximum book size (max common neighbors).
    3. Efficiently evaluate swaps by tracking book changes.
    """
    n = 40
    target_edges = 401
    
    # We will try multiple random seeds and initializations to find the best graph
    best_max_book = 100
    best_adj = None
    
    # Time limit check helper
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(42 + attempt)
        
        # Step 1: Initialize with a random graph
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        edges_set = set(all_edges[:target_edges])
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in edges_set:
            adj[u, v] = 1
            adj[v, u] = 1
            
        # Helper to compute max book from adjacency matrix
        def get_max_book(adj):
            A2 = adj @ adj
            max_b = 0
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v]:
                        if A2[u, v] > max_b:
                            max_b = A2[u, v]
            return max_b, A2

        current_max, A2 = get_max_book(adj)
        
        # Local search
        max_iterations = 500
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
                
            # If we reached the theoretical lower bound, we can stop early for this attempt
            if current_max == 7:
                break
                
            # Find edges with max book
            high_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] and A2[u, v] == current_max:
                        high_edges.append((u, v))
            
            if not high_edges:
                break
                
            random.shuffle(high_edges)
            improved = False
            
            # Try to fix the worst edges
            for (u, v) in high_edges[:5]:
                # We want to remove (u,v) and add (w,x)
                # Removing (u,v) reduces the book size of (u,k) and (v,k) for k in N(u,v)
                # Adding (w,x) increases the book size of (w,k) and (x,k) for k in N(w,x)
                
                # Calculate potential benefit of removing (u,v)
                # The book size of (u,v) is A2[u,v].
                # After removal, A2[u,v] becomes 0 (since edge is gone, book is undefined/0 for that pair, but we care about other edges).
                # Actually, book size is defined for edges. If we remove (u,v), it's no longer an edge.
                # The edges that are affected are those incident to u or v.
                
                # Let's just try a few random swaps involving (u,v) and see if max decreases
                # To be efficient, we can pick a random non-edge (w,x) and check.
                
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
                random.shuffle(non_edges)
                
                found_swap = False
                for w, x in non_edges[:20]: # Try 20 random non-edges
                    # Perform swap
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[w, x] = 1
                    adj[x, w] = 1
                    
                    new_max, new_A2 = get_max_book(adj)
                    
                    if new_max < current_max:
                        current_max = new_max
                        A2 = new_A2
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Revert
                        adj[u, v] = 1
                        adj[v, u] = 1
                        adj[w, x] = 0
                        adj[x, w] = 0
                
                if found_swap:
                    break # Restart high_edges loop with new state
            
            if not improved:
                # Random perturbation to escape local minima
                if iteration % 10 == 9:
                    edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
                    if edge_list:
                        u, v = random.choice(edge_list)
                        adj[u, v] = 0
                        adj[v, u] = 0
                        
                        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
                        if non_edges:
                            w, x = random.choice(non_edges)
                            adj[w, x] = 1
                            adj[x, w] = 1
                            
                            current_max, A2 = get_max_book(adj)
        
        if current_max < best_max_book:
            best_max_book = current_max
            best_adj = adj.copy()
            
            if best_max_book == 7:
                break # Can't do better than 7 theoretically
        
    adj = best_adj
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                result.append([u, v])
                
    if len(result) != target_edges:
        # Fallback: just return a random valid graph if something went wrong
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        result = [list(e) for e in all_edges[:target_edges]]
        
    return result
# EVOLVE-BLOCK-END
