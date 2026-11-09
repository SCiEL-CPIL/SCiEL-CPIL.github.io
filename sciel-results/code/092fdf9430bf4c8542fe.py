import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n = 40
    target_edges = 401
    
    # 1. Initialization: Random graph with target edges
    # We start with a random graph to avoid getting stuck in a poor local minimum
    # specific to the K_{20,20} structure which has high book sizes.
    
    edges = set()
    while len(edges) < target_edges:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Setup Data Structures
    # adj_mask[i] is a numpy array of length n, where adj_mask[i][j] is 1 if (i,j) is an edge.
    adj_mask = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj_mask[u, v] = 1
        adj_mask[v, u] = 1
        
    # book_size[u, v] stores the number of common neighbors (triangles containing edge u-v)
    # We only care about this for existing edges.
    book_size = np.zeros((n, n), dtype=np.int32)
    
    def update_book_size(u, v):
        """Calculate and update book_size[u, v] and book_size[v, u]"""
        if u == v:
            return
        # Count common neighbors using bitwise AND and sum
        count = np.sum(adj_mask[u] & adj_mask[v])
        book_size[u, v] = count
        book_size[v, u] = count
        
    # Initialize book sizes for all edges
    for u, v in edges:
        update_book_size(u, v)
        
    # Track the maximum book size
    max_book = 0
    for u, v in edges:
        if book_size[u, v] > max_book:
            max_book = book_size[u, v]
            
    # 3. Local Search
    # We will perform a best-first search, always trying to fix the edge with the highest book size.
    # If we can't improve the max, we might accept neutral moves to escape local optima.
    
    best_max_book = max_book
    best_edges = set(edges)
    
    # Precompute list of all possible edges for random sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Helper to get a random non-edge
    def get_random_non_edge():
        while True:
            u, v = random.choice(all_pairs)
            if (u, v) not in edges:
                return u, v
                
    # Helper to get the edge with max book size
    def get_max_edge():
        max_b = -1
        max_e = None
        for u, v in edges:
            b = book_size[u, v]
            if b > max_b:
                max_b = b
                max_e = (u, v)
        return max_e, max_b

    no_improve_counter = 0
    
    while time.time() < deadline:
        # Find the current worst edge
        worst_edge, current_max = get_max_edge()
        
        if current_max < best_max_book:
            best_max_book = current_max
            best_edges = set(edges)
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            
        # If we've found a very low book size (e.g., 3 or 4), we can be more selective
        # But for now, we just try to improve.
        
        if worst_edge is None:
            break
            
        u, v = worst_edge
        worst_book_val = book_size[u, v]
        
        # Try to swap (u, v) with a random non-edge (x, y)
        # We limit the number of attempts per iteration to keep the loop fast
        attempts = 0
        max_attempts = 50
        
        improved = False
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            x, y = get_random_non_edge()
            
            # Check if swap is valid (it is, by construction of non-edge)
            # Perform swap
            edges.remove((u, v))
            edges.add((x, y))
            
            # Update adj masks
            adj_mask[u, v] = 0
            adj_mask[v, u] = 0
            adj_mask[x, y] = 1
            adj_mask[y, x] = 1
            
            # Update book sizes for edges affected by the change
            # Edges affected are those incident to u, v, x, y
            affected_nodes = {u, v, x, y}
            
            # We need to recalculate book_size for all edges incident to these nodes
            # Note: book_size is symmetric, so we only need to update one direction? 
            # No, we update both to keep consistency, or just rely on the fact that 
            # we read book_size[a, b] where (a,b) is in edges.
            # Since we iterate over edges, we just need to ensure book_size is correct for existing edges.
            
            # Edges to update:
            # 1. Edges incident to u (excluding v, which is removed)
            # 2. Edges incident to v (excluding u, which is removed)
            # 3. Edges incident to x
            # 4. Edges incident to y
            
            # Optimization: Instead of updating all, we can just calculate the new max.
            # But we need accurate book_size for future iterations.
            # Let's update the specific edges.
            
            # Helper to update an edge's book size
            def upd(a, b):
                if (a, b) in edges:
                    update_book_size(a, b)
            
            # Update edges connected to u
            for neighbor in range(n):
                if neighbor == v: continue
                if (u, neighbor) in edges:
                    upd(u, neighbor)
                    
            # Update edges connected to v
            for neighbor in range(n):
                if neighbor == u: continue
                if (v, neighbor) in edges:
                    upd(v, neighbor)
                    
            # Update edges connected to x
            for neighbor in range(n):
                if (x, neighbor) in edges:
                    upd(x, neighbor)
                    
            # Update edges connected to y
            for neighbor in range(n):
                if (y, neighbor) in edges:
                    upd(y, neighbor)
            
            # Also, the new edge (x, y) needs its book size
            update_book_size(x, y)
            
            # Calculate new max book size
            new_max = 0
            for a, b in edges:
                if book_size[a, b] > new_max:
                    new_max = book_size[a, b]
            
            # Acceptance Criteria
            if new_max < current_max:
                # Strict improvement
                improved = True
                break
            elif new_max == current_max:
                # Neutral move. Accept with low probability to escape local minima
                # or if we are stuck.
                if no_improve_counter > 100 and random.random() < 0.1:
                    improved = True
                    break
            else:
                # Revert
                edges.remove((x, y))
                edges.add((u, v))
                
                adj_mask[x, y] = 0
                adj_mask[y, x] = 0
                adj_mask[u, v] = 1
                adj_mask[v, u] = 1
                
                # Revert book sizes
                for neighbor in range(n):
                    if neighbor == v: continue
                    if (u, neighbor) in edges:
                        upd(u, neighbor)
                        
                for neighbor in range(n):
                    if neighbor == u: continue
                    if (v, neighbor) in edges:
                        upd(v, neighbor)
                        
                for neighbor in range(n):
                    if (x, neighbor) in edges:
                        upd(x, neighbor)
                        
                for neighbor in range(n):
                    if (y, neighbor) in edges:
                        upd(y, neighbor)
                        
                update_book_size(x, y) # x,y is not in edges now, but we removed it. 
                # Actually, we should revert the book_size of the removed edge (u,v) too?
                # No, we don't care about book_size of non-edges.
                # But we need to restore book_size for edges that were updated.
                # The loop above does that.
                
                # We also need to restore the book_size for (u,v) if it was updated?
                # It was removed, so we don't need it.
                # However, when we re-add (u,v), we should update its book size?
                # It will be updated in the next iteration if it's the worst edge.
                # Or we can update it now.
                update_book_size(u, v)
                
                # Revert new edge book size? It's gone.
                
                continue

        if not improved:
            # If we couldn't improve in this batch, we might be stuck.
            # We can try a random restart or just continue.
            # Given the time constraint, continuing with random swaps is better.
            pass

    # Return the best solution found
    result = [[u, v] for u, v in best_edges]
    return result
