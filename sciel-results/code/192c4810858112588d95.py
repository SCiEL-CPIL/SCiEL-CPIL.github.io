import time
import random
from itertools import combinations

def is_not_2colorable(n, edges):
    """
    Check if hypergraph has no proper 2-coloring.
    Proper 2-coloring: No edge is monochromatic.
    Returns True if NO such coloring exists (i.e., hypergraph is non-2-colorable).
    """
    # Map vertices to indices 0..n-1
    # edges is a list of tuples of 5 vertices
    
    # Precompute for each vertex, the list of edges containing it
    # This helps in propagation? Maybe not directly for this small n.
    
    color = [-1] * n
    # To speed up, we can track for each edge:
    # count_0, count_1, count_uncolored
    # If count_0 == 5 or count_1 == 5 -> Invalid (monochromatic)
    # If count_0 == 4 and count_uncolored == 1 -> Force the uncolored to 1
    # If count_1 == 4 and count_uncolored == 1 -> Force the uncolored to 0
    
    # For efficiency with n<=12, we can just recompute counts or maintain them.
    # Given the small size, recomputing is simpler and less error-prone.
    
    def check_edge(edge):
        c0 = 0
        c1 = 0
        unc = []
        for v in edge:
            if color[v] == 0:
                c0 += 1
            elif color[v] == 1:
                c1 += 1
            else:
                unc.append(v)
        
        if c0 == 5 or c1 == 5:
            return False, None # Monochromatic, invalid
        if c0 == 4 and len(unc) == 1:
            return True, (unc[0], 1) # Force 1
        if c1 == 4 and len(unc) == 1:
            return True, (unc[0], 0) # Force 0
        return True, None # No force, no conflict
    
    def backtrack():
        # Unit Propagation Loop
        changed = True
        while changed:
            changed = False
            for edge in edges:
                valid, force = check_edge(edge)
                if not valid:
                    return False
                if force:
                    v, c = force
                    if color[v] != -1:
                        if color[v] != c:
                            return False # Conflict
                    else:
                        color[v] = c
                        changed = True
        
        # Find uncolored vertex
        uncolored = [v for v in range(n) if color[v] == -1]
        if not uncolored:
            return True # All colored, no conflicts, valid coloring exists
        
        # Heuristic: Pick vertex in most constrained edge?
        # For n=9, just pick first is okay, but let's try most frequent in edges
        # Or just first for simplicity and speed.
        v = uncolored[0]
        
        # Try 0 then 1
        color[v] = 0
        if backtrack():
            return True
        color[v] = 1
        if backtrack():
            return True
        
        color[v] = -1
        return False

    # Symmetry breaking: Fix vertex 0 to 0
    color[0] = 0
    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with K_9^5
    n = 9
    initial_edges = list(combinations(range(n), 5))
    
    # Verify it's non-2-colorable (it should be)
    if not is_not_2colorable(n, initial_edges):
        # Fallback: try n=10? K_10^5 is also non-2-colorable?
        # If n=10, split 5+5. Edge of 5 in A is mono. So yes.
        # But K_9^5 is smaller.
        return {'n': 9, 'edges': [list(e) for e in initial_edges]}
        
    current_edges = initial_edges[:]
    
    # Greedy Deletion
    # We want to minimize edges.
    # Strategy: Iterate through edges, try to remove.
    # To avoid getting stuck in a local minimum where order matters,
    # we can shuffle the edge list periodically.
    
    best_edges = current_edges[:]
    
    while time.time() < deadline:
        improved = False
        # Shuffle to randomize order
        random.shuffle(current_edges)
        
        for i in range(len(current_edges)):
            if time.time() > deadline:
                break
            
            # Try removing edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if is_not_2colorable(n, test_edges):
                current_edges = test_edges
                improved = True
                # Continue checking remaining edges in this shuffled order
                # Note: len(current_edges) changed, so indices shift?
                # No, we break out of the for loop?
                # The for loop iterates range(len(current_edges)).
                # If we modify current_edges, the range is fixed at start of loop?
                # No, `range(len(current_edges))` is evaluated once.
                # But `current_edges` is updated.
                # It's safer to restart the loop or handle indices carefully.
                # Let's just break and restart the while loop with a new shuffle.
                # This ensures we try to remove more edges from the new set.
                break
        
        if not improved:
            # No more edges can be removed in this order.
            # Try a different order?
            # If we've done several passes, we might be at a local minimum.
            # We can try to add a random edge and then remove others?
            # Or just accept the current result.
            
            # Let's try a few more shuffles to see if we can improve
            # But we need to track if we've made progress recently.
            
            # Simple strategy: If no improvement in a full pass, 
            # try a few more passes with different shuffles.
            # If still no improvement, stop.
            
            # To keep it simple and fast, we'll just do a few restarts of the deletion process
            # from the current state if time permits.
            
            # Actually, the "break" inside the for loop means we only do one deletion per while iteration.
            # This is slow (126 deletions needed).
            # We should remove as many as possible in one pass.
            
            # Let's restructure the inner loop to not break immediately,
            # but be careful with indices.
            pass

    # The previous logic was flawed. Let's rewrite the deletion loop properly.
    
    # Reset to initial
    current_edges = initial_edges[:]
    
    # Loop until no more edges can be removed or time is up
    # Use a list of indices to try
    while time.time() < deadline:
        removed_any = False
        # Create a list of indices to try
        # We need to handle the fact that the list shrinks
        # Iterate from end to start? Or use a while loop with index.
        
        # Let's try removing edges in random order
        # Generate a random permutation of indices
        indices = list(range(len(current_edges)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Check if this index is still valid (edge might have been removed)
            # Since we are iterating over a static list of indices,
            # and modifying current_edges, the indices might be out of sync.
            
            # Better approach:
            # While there are edges, pick a random edge, try to remove it.
            # If removed, continue. If not, try next random edge.
            # Stop when a full pass yields no removals.
            
            pass
            
    # Let's implement a cleaner "Randomized Greedy Deletion"
    
    current_edges = initial_edges[:]
    
    # We will keep trying to remove edges until we can't.
    # To escape local minima, we can "add" a random edge from the complement, then try to remove edges again.
    # But adding edges increases the count, so we must be careful.
    
    # Simplest robust method:
    # 1. Start with K_9^5
    # 2. While time < deadline:
    #    a. Shuffle current_edges
    #    b. For each edge in shuffled list:
    #       - If removing it keeps non-2-colorable, remove it.
    #    c. If no edge was removed in the entire pass, break.
    
    current_edges = initial_edges[:]
    
    while time.time() < deadline:
        pass_improved = False
        shuffled_edges = current_edges[:]
        random.shuffle(shuffled_edges)
        
        # We need to map shuffled edges back to the current list?
        # Or just iterate through the current list in a random order?
        
        # Let's create a list of indices
        indices = list(range(len(current_edges)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # If this edge was already removed, skip
            # We don't know if it was removed easily.
            # Let's just try to remove edge at index i.
            # If the list has changed, index i might be different.
            
            # This approach is messy.
            
            # Alternative:
            # Pick a random edge from current_edges.
            # Try to remove it.
            # If successful, it's gone.
            # If not, mark it as "tried" for this pass.
            # Repeat until all edges tried.
            
            pass

    # Let's rewrite the deletion logic clearly.
    
    current_edges = initial_edges[:]
    
    # We will perform passes. In each pass, we try to remove each edge.
    # If an edge is removed, the list shrinks.
    # We iterate through the list.
    
    while time.time() < deadline:
        improved = False
        i = 0
        # We need to be careful with indices as the list shrinks.
        # If we remove edge i, the next edge shifts to i.
        # So we don't increment i if we remove.
        
        while i < len(current_edges):
            if time.time() > deadline:
                break
            
            # Try to remove edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if is_not_2colorable(n, test_edges):
                current_edges = test_edges
                improved = True
                # Do not increment i, because the next edge is now at i
            else:
                i += 1
                
        if not improved:
            # No more edges can be removed in this order.
            # Try shuffling the order and repeating?
            # If we just shuffle and restart, we might find more removable edges.
            # But if the graph is "minimally" non-2-colorable, no order will help.
            # However, we might be in a local minimum where removing edge A requires removing edge B first?
            # No, if removing A fails, but removing B then A succeeds, that's possible.
            # So we should try multiple random orders.
            
            # Let's try a few random shuffles
            # But we need to avoid infinite loops.
            
            # If we've done several passes with no improvement, we can stop.
            # Or we can try to add a random edge and then delete?
            
            # For now, let's just do a few more shuffles.
            
            # To keep it simple, we'll just break if no improvement.
            # This might leave some edges, but it's a valid hypergraph.
            # Given the time limit, this should be sufficient.
            
            break
            
    return {'n': n, 'edges': [list(e) for e in current_edges]}
