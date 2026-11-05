import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 40
    target_edges = 401
    
    # Adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    
    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        
    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        
    def edge_count():
        return int(np.sum(adj) // 2)
        
    def compute_books():
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    common = np.sum(adj[u] & adj[v])
                    books[(u, v)] = int(common)
        return books
        
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
        
    def objective(books):
        if not books:
            return 0
        vals = list(books.values())
        # Primary: max book. Secondary: sum of squares to flatten distribution
        return max(vals) * 1000 + sum(v*v for v in vals)

    # 1. Initialization: Random graph
    random.seed(42)
    np.random.seed(42)
    
    # Generate random graph with exactly target_edges
    # Start with p=0.5, then adjust
    # Easier: Generate all pairs, shuffle, take first 401
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs[:target_edges]:
        add_edge(u, v)
        
    # Ensure count is correct (it should be)
    if edge_count() != target_edges:
        # Fix if needed
        pass

    books = compute_books()
    best_books = {k: v for k, v in books.items()}
    best_obj = objective(best_books)
    best_adj = adj.copy()
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    start_time = time.time()
    step = 0
    
    # Pre-compute list of all possible edges for random selection
    all_edges_list = all_pairs
    
    while T > T_min and time.time() < deadline:
        step += 1
        
        # Targeted Move:
        # 1. Find edges with book size > 7
        # 2. Pick one randomly
        # 3. Try to remove it and add a different edge
        
        bad_edges = [e for e, b in books.items() if b > 7]
        
        if not bad_edges:
            # If no bad edges, try to improve the secondary objective (sum of squares)
            # Or just do random swaps to explore
            # Pick a random edge to remove
            current_edges = list(books.keys())
            if not current_edges:
                break
            u1, v1 = random.choice(current_edges)
        else:
            u1, v1 = random.choice(bad_edges)
            
        # Try to find a good swap
        # Remove (u1, v1)
        # Add (u2, v2)
        
        # We will try a few candidates for (u2, v2)
        # To speed up, we can pre-calculate potential gains?
        # For now, random candidates.
        
        best_candidate = None
        best_candidate_obj = objective(books)
        
        # Try 50 random candidates
        for _ in range(50):
            u2, v2 = random.choice(all_edges_list)
            if (u2, v2) == (u1, v1):
                continue
            if adj[u2, v2]:
                continue
            
            # Calculate change
            # Remove (u1, v1), Add (u2, v2)
            
            # Affected edges: incident to u1, v1, u2, v2
            affected = set()
            for x in [u1, v1, u2, v2]:
                for y in range(n):
                    if adj[x, y]:
                        a, b = (x, y) if x < y else (y, x)
                        affected.add((a, b))
            affected.add((u2, v2))
            
            old_books_affected = {e: books[e] for e in affected if e in books}
            
            # Apply
            remove_edge(u1, v1)
            add_edge(u2, v2)
            
            # Update books for affected
            for e in affected:
                if e in books:
                    del books[e]
            
            new_books_affected = {}
            for e in affected:
                a, b = e
                if adj[a, b]:
                    common = int(np.sum(adj[a] & adj[b]))
                    new_books_affected[e] = common
                    books[e] = common
            
            new_obj = objective(books)
            
            if new_obj < best_candidate_obj:
                best_candidate_obj = new_obj
                best_candidate = (u2, v2, new_books_affected, old_books_affected, affected)
            
            # Undo
            remove_edge(u2, v2)
            add_edge(u1, v1)
            
            for e in affected:
                if e in books:
                    del books[e]
            for e, val in old_books_affected.items():
                books[e] = val
                
        if best_candidate:
            u2, v2, new_books_affected, old_books_affected, affected = best_candidate
            
            # Apply the best candidate
            remove_edge(u1, v1)
            add_edge(u2, v2)
            
            for e in affected:
                if e in books:
                    del books[e]
            for e, val in new_books_affected.items():
                books[e] = val
                
            new_obj = objective(books)
            delta = new_obj - best_obj
            
            # Accept if better or with prob exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                best_obj = new_obj
                if new_obj < objective(best_books):
                    best_books = {k: v for k, v in books.items()}
                    best_adj = adj.copy()
            else:
                # Reject
                remove_edge(u2, v2)
                add_edge(u1, v1)
                for e in affected:
                    if e in books:
                        del books[e]
                for e, val in old_books_affected.items():
                    books[e] = val
            
        T *= cooling
        
        if step % 1000 == 0:
            if time.time() > deadline - 5.0:
                break
                
    # Restore best
    adj = best_adj
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                result.append([i, j])
                
    return result
