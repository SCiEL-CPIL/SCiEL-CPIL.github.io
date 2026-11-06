import random
import time

def construct():
    n = 40
    m = 401
    
    # Use bitmasks for adjacency: adj[i] is a 40-bit integer where bit j is set if edge (i,j) exists
    # Start with K20,20 + one edge within part A
    # A = 0..19, B = 20..39
    adj = [0] * n
    for u in range(20):
        for v in range(20, 40):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    # Add edge (0,1) within part A
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    # Pre-compute list of all possible edges for random selection
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def get_current_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                v_bit = mask & (-mask)
                v = v_bit.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask ^= v_bit
        return edges

    def compute_max_book(adj, edges):
        """Compute max book size. Iterates only over existing edges."""
        max_book = 0
        for u, v in edges:
            common = (adj[u] & adj[v]).bit_count()
            if common > max_book:
                max_book = common
        return max_book

    current_adj = adj[:]
    current_edges = get_current_edges(current_adj)
    current_max_book = compute_max_book(current_adj, current_edges)
    
    best_adj = current_adj[:]
    best_edges = current_edges[:]
    best_max_book = current_max_book
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Local Search with Random Restarts
    # Strategy: Always try to fix the "worst" edge (max book).
    # If stuck, do random swaps to escape local minima.
    
    while time.time() < deadline:
        # Find the edge with maximum book size in current state
        max_edge = None
        max_book_val = 0
        for u, v in current_edges:
            common = (current_adj[u] & current_adj[v]).bit_count()
            if common > max_book_val:
                max_book_val = common
                max_edge = (u, v)
        
        if max_edge is None:
            break
            
        u, v = max_edge
        
        # Try to remove the worst edge and add a random non-existent edge
        # We prefer adding edges that don't create large books.
        # Heuristic: Randomly pick a non-edge.
        # To be efficient, we just pick a random possible edge and check if it exists.
        
        improved = False
        # Try a few targeted moves
        for _ in range(20):
            # Pick a random edge to add
            add_u, add_v = random.choice(all_possible_edges)
            
            # Check if it's already in the graph
            if (current_adj[add_u] & (1 << add_v)):
                continue
                
            # Check if it's the same as the one we're removing (shouldn't happen if we remove first, but safe)
            if (add_u, add_v) == (u, v) or (add_u, add_v) == (v, u):
                continue

            # Apply move
            # Remove (u, v)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Add (add_u, add_v)
            new_adj[add_u] |= (1 << add_v)
            new_adj[add_v] |= (1 << add_u)
            
            # Update edges list for evaluation
            # Instead of rebuilding the whole list, we can just evaluate the new max book.
            # But we need the list of edges for the next iteration.
            # For speed, let's just compute the max book on the new_adj using the updated edge set.
            # Rebuilding the edge list is O(m) ~ 400. Computing max book is O(m) ~ 400.
            # This is fast enough.
            
            new_edges = get_current_edges(new_adj)
            new_max_book = compute_max_book(new_adj, new_edges)
            
            if new_max_book < current_max_book:
                # Accept move
                current_adj = new_adj
                current_edges = new_edges
                current_max_book = new_max_book
                improved = True
                
                if new_max_book < best_max_book:
                    best_max_book = new_max_book
                    best_adj = new_adj
                    best_edges = new_edges
                break # Restart search from new state
            
        if not improved:
            # If we couldn't improve by fixing the worst edge, try random swaps
            # This helps escape local minima
            for _ in range(10):
                # Pick a random edge to remove
                rem_u, rem_v = random.choice(current_edges)
                # Pick a random edge to add
                add_u, add_v = random.choice(all_possible_edges)
                
                if (add_u, add_v) == (rem_u, rem_v) or (add_u, add_v) == (rem_v, rem_u):
                    continue
                if (current_adj[add_u] & (1 << add_v)):
                    continue

                new_adj = current_adj[:]
                new_adj[rem_u] &= ~(1 << rem_v)
                new_adj[rem_v] &= ~(1 << rem_u)
                
                new_adj[add_u] |= (1 << add_v)
                new_adj[add_v] |= (1 << add_u)
                
                new_edges = get_current_edges(new_adj)
                new_max_book = compute_max_book(new_adj, new_edges)
                
                # Accept if better or equal (to allow wandering)
                if new_max_book <= current_max_book:
                    current_adj = new_adj
                    current_edges = new_edges
                    current_max_book = new_max_book
                    
                    if new_max_book < best_max_book:
                        best_max_book = new_max_book
                        best_adj = new_adj
                        best_edges = new_edges
                    break

    # Convert best_adj to list of edges
    result = []
    for u, v in best_edges:
        result.append([u, v])
    
    return result
