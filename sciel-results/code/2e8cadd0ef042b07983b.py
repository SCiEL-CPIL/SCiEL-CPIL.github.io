"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
# EVOLVE-BLOCK-START

def _has_2_coloring(n, edges):
    """
    Check if the hypergraph (n vertices, 5-uniform edges) has a proper 2-coloring.
    Uses backtracking with unit propagation.
    Returns True if a proper 2-coloring exists, False otherwise.
    """
    if not edges:
        return True
        
    m = len(edges)
    # Precompute adjacency: vertex -> list of edge indices
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    # State:
    # assignment[v]: -1 (unassigned), 0, 1
    # edge_mask[i]: bitmask of colors present (1 for 0, 2 for 1)
    # edge_unassigned[i]: count of unassigned vertices in edge i
    
    assignment = [-1] * n
    edge_mask = [0] * m
    edge_unassigned = [5] * m
    
    # Optimization: If any edge has 0 unassigned vertices (impossible at start) 
    # or if we detect a contradiction early.
    
    def propagate(assignment, edge_mask, edge_unassigned):
        """
        Apply unit propagation.
        Returns True if consistent, False if contradiction.
        Modifies assignment, edge_mask, edge_unassigned in place.
        """
        # We use a queue of vertices that have been assigned to trigger updates.
        # But since we check all edges in a loop, we can just iterate.
        # For small n and m, a simple loop is fast enough.
        
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if edge_unassigned[i] == 0:
                    if edge_mask[i] != 3:
                        return False
                    continue
                    
                if edge_unassigned[i] == 1:
                    # Find the unassigned vertex in this edge
                    unassigned_v = -1
                    for v in edges[i]:
                        if assignment[v] == -1:
                            unassigned_v = v
                            break
                            
                    if unassigned_v == -1:
                        continue # Should not happen if counts are correct
                        
                    if edge_mask[i] == 1: # Only 0s seen
                        # Must be 1
                        assignment[unassigned_v] = 1
                        edge_unassigned[i] -= 1
                        edge_mask[i] |= 2
                        # Update other edges containing unassigned_v
                        for j in vertex_to_edges[unassigned_v]:
                            if j != i:
                                edge_mask[j] |= 2
                                edge_unassigned[j] -= 1
                        changed = True
                    elif edge_mask[i] == 2: # Only 1s seen
                        # Must be 0
                        assignment[unassigned_v] = 0
                        edge_unassigned[i] -= 1
                        edge_mask[i] |= 1
                        # Update other edges containing unassigned_v
                        for j in vertex_to_edges[unassigned_v]:
                            if j != i:
                                edge_mask[j] |= 1
                                edge_unassigned[j] -= 1
                        changed = True
                    else:
                        # Already satisfied (mask 3), nothing to force
                        pass
                        
            # Check for contradictions after propagation step
            for i in range(m):
                if edge_unassigned[i] == 0 and edge_mask[i] != 3:
                    return False
                    
        return True

    def dfs(assignment, edge_mask, edge_unassigned):
        # Check for immediate contradiction
        for i in range(m):
            if edge_unassigned[i] == 0 and edge_mask[i] != 3:
                return False
                
        # Find unassigned vertices
        unassigned = [v for v in range(n) if assignment[v] == -1]
        if not unassigned:
            # All assigned. Check if all edges satisfied.
            # Since we propagated, if no contradiction was returned, it should be valid?
            # Wait, if an edge has mask 1 and unassigned 0, it's a contradiction.
            # If mask 3, it's satisfied.
            # So if we reach here with no contradiction, it's a valid coloring.
            return True
        
        # Heuristic: Choose vertex with most constraints (highest degree)
        # Or vertex involved in most unsatisfied edges?
        # Simple: max degree
        v = max(unassigned, key=lambda x: len(vertex_to_edges[x]))
        
        # Try color 0
        # Copy state
        assignment2 = assignment[:]
        edge_mask2 = edge_mask[:]
        edge_unassigned2 = edge_unassigned[:]
        
        assignment2[v] = 0
        for j in vertex_to_edges[v]:
            edge_mask2[j] |= 1
            edge_unassigned2[j] -= 1
            
        if propagate(assignment2, edge_mask2, edge_unassigned2):
            if dfs(assignment2, edge_mask2, edge_unassigned2):
                return True
                
        # Try color 1
        assignment3 = assignment[:]
        edge_mask3 = edge_mask[:]
        edge_unassigned3 = edge_unassigned[:]
        
        assignment3[v] = 1
        for j in vertex_to_edges[v]:
            edge_mask3[j] |= 2
            edge_unassigned3[j] -= 1
            
        if propagate(assignment3, edge_mask3, edge_unassigned3):
            if dfs(assignment3, edge_mask3, edge_unassigned3):
                return True
                
        return False

    return dfs(assignment, edge_mask, edge_unassigned)

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Uses a greedy minimization approach starting from a known non-2-colorable hypergraph.
    """
    from itertools import combinations
    import random
    
    # Start with a known non-2-colorable hypergraph: Complete 5-uniform on 9 vertices
    # This has 126 edges.
    n_start = 9
    edges_start = [list(e) for e in combinations(range(n_start), 5)]
    
    # Verify it is non-2-colorable (it is, but good practice)
    # If _has_2_coloring returns True, we have a problem.
    # But K_9^5 is definitely not 2-colorable.
    
    # Greedy Minimization
    # We want to remove edges while preserving non-2-colorability.
    
    def try_minimize(n, edges):
        # Randomize edge order for different local minima
        random.shuffle(edges)
        
        # Repeat a few times to escape local minima
        for _ in range(5):
            changed = True
            while changed:
                changed = False
                # Iterate through edges in random order
                random.shuffle(edges)
                for i in range(len(edges)):
                    # Tentatively remove edge i
                    removed_edge = edges[i]
                    edges.pop(i)
                    
                    # Check if still non-2-colorable
                    if not _has_2_coloring(n, edges):
                        # It is still non-2-colorable, keep it removed
                        changed = True
                        # Continue to next edge (indices shifted, so break inner loop and restart? 
                        # Or just continue carefully. Since we popped, len decreased.
                        # It's safer to restart the while loop or handle indices.
                        # Let's just break and restart the while loop for simplicity and safety.
                        break
                    else:
                        # It became 2-colorable, restore the edge
                        edges.insert(i, removed_edge)
                
                if not changed:
                    break
                    
        return edges

    # Try minimizing the starting hypergraph
    edges_min = try_minimize(n_start, edges_start)
    
    # We can also try other starting points or vertex counts
    # But 9 vertices is small enough that the checker is fast.
    
    # To potentially get fewer edges, we could try starting with a larger n and minimizing,
    # but the edge count might not drop as low.
    # For n=10, we could start with a random subset of K_10^5 that is non-2-colorable.
    # But finding that subset is hard.
    
    # Let's stick to the greedy minimization of the 9-vertex complete hypergraph.
    # This should reduce 126 edges to a much smaller number.
    
    return {'n': n_start, 'edges': edges_min}

# EVOLVE-BLOCK-END
