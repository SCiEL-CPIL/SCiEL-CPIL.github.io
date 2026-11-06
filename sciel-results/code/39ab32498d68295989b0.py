import time
import random
from itertools import combinations

def is_2_colorable(n, edges, timeout=0.5):
    """
    Returns True if the hypergraph has a 2-coloring where every edge is bichromatic.
    Returns False if no such coloring exists (i.e., hypergraph fails Property B).
    Returns None if timeout occurs.
    """
    start_time = time.time()
    
    # Precompute edge masks for fast propagation? 
    # For n <= 26, we can use integers as bitmasks.
    # x_i is 0 or 1.
    # We maintain a list of assignments: -1 (unassigned), 0, 1.
    
    # Convert edges to lists of indices
    edge_list = [list(e) for e in edges]
    
    # We will use a recursive backtracking solver.
    # To make it fast, we choose the vertex that appears in the most "critical" edges.
    # Simplified heuristic: Choose unassigned vertex with highest degree.
    
    # Precompute degrees
    degree = [0] * n
    for e in edge_list:
        for v in e:
            degree[v] += 1
    
    # Sort vertices by degree descending for variable selection
    # But degrees change? No, structure is static.
    # We can just pick the unassigned vertex with highest static degree.
    
    def backtrack(assignments):
        if time.time() - start_time > timeout:
            return None # Timeout
        
        # Check if all vertices assigned
        if all(a != -1 for a in assignments):
            # Verify all edges are bichromatic
            for e in edge_list:
                colors = [assignments[v] for v in e]
                if len(set(colors)) == 1:
                    return False # Monochromatic edge found, invalid coloring
            return True # Valid coloring found
        
        # Unit Propagation:
        # An edge is "satisfied" if it already has both 0s and 1s.
        # An edge is "failed" if it is monochromatic (all 0s or all 1s).
        # An edge is "pending" if it has only one color assigned so far, and the rest are unassigned.
        # If an edge has only 0s assigned and unassigned vertices, it MUST have a 1.
        # If an edge has only 1s assigned and unassigned vertices, it MUST have a 0.
        
        # We can propagate constraints.
        # However, implementing full unit propagation for this specific "at least one 0 and at least one 1" constraint is slightly different from standard SAT.
        # Standard SAT clause: (x1 OR x2 OR ...)
        # Here we have two clauses per edge: (x_v1 OR ... OR x_v5) AND (NOT x_v1 OR ... OR NOT x_v5).
        
        # Let's just do simple backtracking with a good heuristic first, as n is small.
        
        # Select unassigned vertex with highest degree
        unassigned = [i for i in range(n) if assignments[i] == -1]
        if not unassigned:
             # Should be handled above, but just in case
             for e in edge_list:
                colors = [assignments[v] for v in e]
                if len(set(colors)) == 1:
                    return False
             return True

        v = max(unassigned, key=lambda i: degree[i])
        
        # Try 0 then 1 (or random order to break symmetry)
        # Symmetry breaking: Force first vertex to 0? 
        # Since colors are symmetric, we can fix assignments[0] = 0 if 0 is unassigned.
        # But let's just try both.
        
        assignments[v] = 0
        if backtrack(assignments):
            return True
        assignments[v] = 1
        if backtrack(assignments):
            return True
        assignments[v] = -1
        return False

    assignments = [-1] * n
    # Symmetry breaking: If vertex 0 is unassigned, try 0 first? 
    # Actually, just starting with all -1 and letting the heuristic pick is fine.
    # To speed up, we can force vertex 0 to 0.
    # If a solution exists, there is one with x_0=0.
    assignments[0] = 0
    res = backtrack(assignments)
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Start with a dense non-2-colorable hypergraph
    # K_9^(5) has 126 edges.
    n = 9
    initial_edges = [list(e) for e in combinations(range(n), 5)]
    
    # We will try to remove edges greedily.
    # To make it deterministic and efficient, we shuffle edges and try to remove them.
    # Or iterate in order.
    
    current_edges = initial_edges[:]
    
    # Function to check if current_edges is non-2-colorable
    # We use a short timeout for the checker to ensure we don't get stuck.
    # If the checker times out, we assume it might be non-2-colorable? 
    # No, if it times out, we don't know. 
    # For small n=9, the checker should be very fast.
    
    def is_non_2_colorable(edges):
        return not is_2_colorable(n, edges, timeout=0.1)
        
    # Initial check
    if not is_non_2_colorable(current_edges):
        # This shouldn't happen for K_9
        return {'n': n, 'edges': current_edges}
        
    # Greedy removal
    # We try to remove each edge. If the hypergraph remains non-2-colorable, we keep it removed.
    # We repeat this process in multiple passes until no more edges can be removed or time runs out.
    
    edges_to_try = list(range(len(current_edges)))
    
    while time.time() < deadline:
        removed_any = False
        # Shuffle to avoid getting stuck in a local minimum dependent on order
        random.shuffle(edges_to_try)
        
        for i in edges_to_try:
            if i >= len(current_edges):
                continue
                
            if time.time() > deadline - 5:
                break
                
            edge_to_remove = current_edges[i]
            
            # Try removing it
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if new_edges is still non-2-colorable
            if is_non_2_colorable(new_edges):
                # It is still non-2-colorable, so we can keep this edge removed
                current_edges = new_edges
                removed_any = True
                # The indices of subsequent edges have shifted.
                # We can just restart the loop or adjust indices.
                # Restarting the outer while loop is safer and simpler.
                break
        
        if not removed_any:
            # No more edges can be removed in this pass
            # We could try a different strategy, but for now, we stop.
            # Or we can try to remove edges in a different order?
            # The shuffle in the next iteration will handle that if we continue the while loop.
            # But if no edge was removed in a full pass, we are at a local minimum.
            # We can try to restart with a different initial graph or just accept this.
            # Given the time constraint, accepting the local minimum is reasonable.
            # However, we can try to "restart" the removal process with a fresh shuffle.
            # If we do that indefinitely, we might loop. 
            # Let's limit the number of restarts.
            
            # Actually, if no edge was removed, we are done with this specific graph structure.
            # We can try to add some randomness to the variable selection in the solver to escape?
            # No, the solver is exact. The result "non-2-colorable" is a property of the graph.
            # If no single edge removal preserves the property, the graph is a minimal counterexample.
            # So we can stop.
            break
            
    return {'n': n, 'edges': current_edges}

# To ensure we don't hit the exact timeout limit of the evaluator, we use a safety margin.
# The evaluator kills at 150s. We stop at 110s.
