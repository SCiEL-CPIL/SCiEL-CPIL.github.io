import random
import time
import math

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n = 40
    m = 401
    
    # Initial state: K_{20,20} plus one edge (0,1)
    # This structure is known to minimize max degree and triangle density for this n, m.
    # Edges in K_{20,20}: (i, j) where i < 20, j >= 20. Total 20*20 = 400.
    # Plus (0, 1). Total 401.
    
    # Representation:
    # edge_set: set of (min(u,v), max(u,v))
    # adj: list of sets for quick neighbor lookup
    # edge_count: dict mapping edge -> number of triangles containing it (book size)
    
    def init_graph():
        edges = set()
        for i in range(20):
            for j in range(20, 40):
                edges.add((i, j))
        edges.add((0, 1))
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Calculate initial book sizes
        book_sizes = {}
        for u, v in edges:
            # Number of common neighbors
            common = len(adj[u] & adj[v])
            book_sizes[(u, v)] = common
            
        return edges, adj, book_sizes

    edges, adj, book_sizes = init_graph()
    current_max_book = max(book_sizes.values())
    best_edges = set(edges)
    best_score = current_max_book
    
    # Simulated Annealing
    # Move: Remove an edge (u, v) and add a non-edge (x, y).
    # This keeps the number of edges constant at 401.
    
    # Pre-compute all possible edges for sampling
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # To speed up sampling of non-edges, we can just pick random pairs and check.
    # With n=40, total pairs 780. 401 are present. 379 are absent.
    # Probability of picking an absent edge is ~48%.
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Helper to update book sizes efficiently
    # When we remove (u, v) and add (x, y):
    # 1. For every edge (a, b) that shares a vertex with (u, v) or (x, y), its book size might change.
    # Actually, the book size of an edge (a, b) is the number of common neighbors.
    # Removing (u, v) affects book sizes of edges (u, w) for w in adj[u] and (v, w) for w in adj[v].
    # Adding (x, y) affects book sizes of edges (x, w) for w in adj[x] and (y, w) for w in adj[y].
    # Also, edges (u, x), (u, y), (v, x), (v, y) might become triangles or cease to be.
    
    # Given n=40 and m=401, degrees are ~20.
    # Updating neighbors of 4 vertices involves ~80 edges.
    # Recalculating book size for an edge takes O(degree) or O(1) if we maintain counts.
    # Let's just recalculate book sizes for affected edges.
    
    def get_affected_edges(removed_edges, added_edges):
        affected = set()
        for u, v in removed_edges:
            for w in adj[u]:
                affected.add((min(u, w), max(u, w)))
            for w in adj[v]:
                affected.add((min(v, w), max(v, w)))
        for u, v in added_edges:
            for w in adj[u]:
                affected.add((min(u, w), max(u, w)))
            for w in adj[v]:
                affected.add((min(v, w), max(v, w)))
        return affected

    # We will do a simple restart-based SA or just one long SA.
    # Given the time limit, a long SA with periodic restarts if stuck might be good.
    # But let's try one long SA first.
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(list(edges))
        
        # Pick a random non-edge to add
        # Retry a few times if we hit an edge
        for _ in range(10):
            x, y = random.sample(range(n), 2)
            if x > y: x, y = y, x
            if (x, y) not in edges:
                break
        else:
            continue # Couldn't find a non-edge? Unlikely with 379 non-edges.
            
        # Proposed move: remove (u, v), add (x, y)
        # Check if (x, y) is already connected? No, it's a non-edge.
        
        # Calculate delta in max book size?
        # It's hard to calculate delta in MAX book size directly without updating all.
        # We can estimate the cost as the max book size of the added edge (x, y) 
        # and the removed edge (u, v).
        
        # Book size of (x, y) if added:
        # It will be len(adj[x] & adj[y]) + (1 if u==x or u==y or v==x or v==y else 0 ... wait)
        # Actually, adj sets don't include (x, y) yet.
        # Common neighbors of x and y in current graph:
        common_xy = len(adj[x] & adj[y])
        # If we add (x, y), does it create new triangles for (x, y)?
        # Yes, for every common neighbor w, (x, y, w) is a triangle.
        # So new book size of (x, y) is common_xy.
        
        # What about other edges?
        # Edges sharing x or y will have their book size changed.
        # Specifically, if (x, w) is an edge, its book size changes by +/- 1 depending on if y was a common neighbor.
        # This is complex.
        
        # Simpler approach:
        # 1. Perform the swap.
        # 2. Update adj sets.
        # 3. Recalculate book sizes for affected edges.
        # 4. Determine new max book.
        # 5. Accept/Reject.
        # 6. Undo if rejected.
        
        # To make this fast, we only recalculate book sizes for edges incident to u, v, x, y.
        # Let's collect the set of edges to update.
        edges_to_update = set()
        
        # Edges incident to u
        for w in adj[u]:
            edges_to_update.add((min(u, w), max(u, w)))
        # Edges incident to v
        for w in adj[v]:
            edges_to_update.add((min(v, w), max(v, w)))
        # Edges incident to x
        for w in adj[x]:
            edges_to_update.add((min(x, w), max(x, w)))
        # Edges incident to y
        for w in adj[y]:
            edges_to_update.add((min(y, w), max(y, w)))
            
        # Also (u, v) itself is being removed, so it's not in book_sizes after removal.
        # (x, y) is being added, so it needs to be in book_sizes.
        
        # Save old book sizes for these edges
        old_books = {}
        for e in edges_to_update:
            if e in book_sizes:
                old_books[e] = book_sizes[e]
        
        # Perform Swap
        edges.remove((u, v))
        edges.add((x, y))
        
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].add(y)
        adj[y].add(x)
        
        # Recalculate book sizes for edges_to_update
        # Note: (u, v) is no longer an edge, so we don't care about its book size.
        # (x, y) is now an edge.
        new_max = 0
        for e in edges_to_update:
            if e in edges:
                a, b = e
                common = len(adj[a] & adj[b])
                book_sizes[e] = common
                if common > new_max:
                    new_max = common
            else:
                # Edge was removed (only u, v should be removed from this set potentially, 
                # but we only added edges incident to u, v, x, y. 
                # If e == (u, v), it is not in edges.
                # If e is some other edge, it should still be in edges unless it was (u, v).
                pass
        
        # We need to know the global max book size.
        # We only updated a subset. The max of the updated subset might be lower than the old global max
        # if the old global max was in an unupdated edge.
        # However, the old global max was `current_max_book`.
        # If the edge that had `current_max_book` is in `edges_to_update`, we know the new value.
        # If it is NOT in `edges_to_update`, its value hasn't changed, so it's still `current_max_book`.
        
        # So, new_global_max = max(new_max, current_max_book) 
        # WAIT. If the edge with current_max_book is updated, its new value is in new_max (if it's the max of the updated set) or lower.
        # But other edges not updated still have their old values.
        # So the new global max is max( max(book_sizes[e] for e in edges if e not in edges_to_update), new_max )
        
        # We don't want to iterate all 401 edges every step.
        # Heuristic: 
        # 1. If (u, v) was the unique edge with current_max_book, and we removed it, the new max might drop.
        # 2. If we added (x, y) with high book size, new max might rise.
        
        # To be safe and simple, let's just track the max of the updated edges and assume the rest are <= current_max_book.
        # This is valid if current_max_book was the max of ALL edges.
        # new_global_max = max(new_max, current_max_book) is WRONG if the edge with current_max_book was updated and its value dropped.
        
        # Let's identify if the "critical" edge was updated.
        # We don't track which edge had the max.
        
        # Alternative: Just recompute global max every N steps? Or just accept if new_max <= current_max_book?
        # If new_max <= current_max_book, it's possible the global max stayed the same (if an unupdated edge had it) or dropped (if the max edge was updated and dropped).
        # If new_max > current_max_book, the global max definitely increased.
        
        # Let's use a stricter acceptance:
        # If new_max > current_max_book:
        #    delta = new_max - current_max_book
        #    Accept with prob exp(-delta/T)
        # Else:
        #    # It might have improved or stayed same.
        #    # To know for sure, we'd need to check if the old max edge is still max.
        #    # For now, let's assume if new_max <= current_max_book, we accept (optimistic) or check specifically.
        #    # Let's just accept if new_max <= current_max_book. This allows exploration.
        #    # If we get stuck, we can add a check.
        
        if new_max > current_max_book:
            delta = new_max - current_max_book
            if random.random() < math.exp(-delta / T):
                # Accept
                current_max_book = new_max
                if current_max_book < best_score:
                    best_score = current_max_book
                    best_edges = set(edges)
            else:
                # Reject: Undo
                edges.add((u, v))
                edges.remove((x, y))
                
                adj[u].add(v)
                adj[v].add(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                # Restore book sizes
                for e, val in old_books.items():
                    book_sizes[e] = val
                # Remove (x, y) from book_sizes if it was added
                if (x, y) in book_sizes:
                    del book_sizes[(x, y)]
        else:
            # new_max <= current_max_book
            # It's an improvement or neutral. Accept.
            # But wait, did the global max actually drop?
            # If the edge with current_max_book was in edges_to_update, its new value is <= new_max <= current_max_book.
            # If it wasn't, it's still current_max_book.
            # So the new global max is max(new_max, current_max_book) if the old max edge wasn't updated?
            # No, if the old max edge WAS updated, its new value is in new_max.
            # If the old max edge WASN'T updated, its value is current_max_book.
            # So the new global max is either new_max (if old max edge was updated and dropped) or current_max_book (if not).
            # In either case, it is <= current_max_book.
            # So we can safely set current_max_book = max(new_max, current_max_book)?
            # No, if the old max edge was updated and dropped to 5, and new_max is 5, but another unupdated edge has 7 (which was < old max 8), then new global max is 7.
            # We don't know the second highest.
            
            # To be robust, let's just accept the move.
            # We will periodically recompute the true global max to correct drift.
            current_max_book = max(new_max, current_max_book) # This is an upper bound, might be loose.
            
            # Actually, let's just recompute global max every 100 steps to be safe.
            # Or, simpler: just accept. The score is the max book size.
            # If we accept a move that doesn't reduce the max, we haven't improved, but we haven't worsened it (assuming our upper bound logic holds).
            # Let's just track best_score by recomputing when we think we improved?
            # No, let's just recompute global max every 50 steps.
            
            if not hasattr(construct, 'step_count'):
                construct.step_count = 0
            construct.step_count += 1
            
            if construct.step_count % 50 == 0:
                true_max = 0
                for e in edges:
                    a, b = e
                    if book_sizes.get(e, 0) > true_max:
                        true_max = book_sizes[e]
                    # If book_sizes is stale, recompute?
                    # We updated book_sizes for affected edges.
                    # Unaffected edges have correct book_sizes.
                    # So iterating all edges and taking max of book_sizes[e] is correct.
                current_max_book = true_max
                
                if current_max_book < best_score:
                    best_score = current_max_book
                    best_edges = set(edges)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Final check
    if best_edges is None:
        best_edges = edges
        
    # Convert to list of lists
    return [[u, v] for u, v in best_edges]

# Reset step count for multiple calls if any (though usually called once)
construct.step_count = 0
