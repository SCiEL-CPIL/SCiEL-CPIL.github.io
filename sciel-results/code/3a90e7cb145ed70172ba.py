<<<<<<< SEARCH
def is_property_b(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring.
    Returns True if Property B holds (no 2-coloring).
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Map vertices to a set of incident edges
    vertex_edges = [set() for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].add(i)
            
    # We want to assign colors 0 or 1 to vertices.
    # An edge is "satisfied" (not monochromatic) if it has at least one 0 and one 1.
    # "No proper 2-coloring" means for EVERY assignment, there exists an edge that is monochromatic.
    # Equivalently, the system of constraints "for all edges, not all vertices same color" is unsatisfiable.
    
    # Backtracking with unit propagation on the CNF:
    # For each edge {v1, ..., v5}, the constraint is:
    # (v1 OR ... OR v5) AND (!v1 OR ... OR !v5)
    # This is a 2-CNF? No, it's a 5-CNF.
    # Unit propagation works if we fix some variables.
    
    # Simpler: Just use a standard DPLL/CDCL-lite or simple backtracking.
    # Given n <= 26, simple backtracking might be slow if not careful.
    # But we only need to check a few candidates.
    
    # Let's use a recursive backtracking with pruning.
    # State: assignment array of size n, -1 for unassigned.
    # Pruning: If an edge becomes monochromatic with the current partial assignment (and the rest are unassigned or same color), we can prune?
    # No, if an edge is currently all 0s, it's satisfied. If it's all 0s and 1s, it's satisfied.
    # If it's all 0s and the rest are unassigned, it *could* become unsatisfied if all remaining become 0? No, if all become 0, it IS monochromatic (bad for coloring, good for Property B).
    # We are looking for a coloring that satisfies ALL edges (i.e., no edge is monochromatic).
    # If we find such a coloring, Property B does NOT hold.
    # If no such coloring exists, Property B holds.
    
    # So, we search for a valid 2-coloring. If we find one, return False. If search space exhausted, return True.
    
    def backtrack(assign):
        # Check if current assignment is valid so far (no edge is already monochromatic with all assigned vertices)
        # Actually, if an edge has all vertices assigned and they are all same color, it's a failure for the coloring.
        # If an edge has all vertices assigned and they are mixed, it's OK.
        # If an edge has some assigned, some unassigned:
        # If all assigned are 0, and if all unassigned become 0, it fails. But we don't know yet.
        # Unit propagation:
        # If an edge has 4 vertices assigned as 0, and 1 unassigned, the unassigned MUST be 1 to avoid monochromatic 0.
        # If an edge has 4 vertices assigned as 1, and 1 unassigned, the unassigned MUST be 0.
        
        # Let's implement a simple backtracking with this propagation.
        
        # For efficiency, we can just check all edges at each step.
        # n is small (<= 26), edges <= 200.
        
        # Find first unassigned vertex
        v = -1
        for i in range(n):
            if assign[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned. Check if all edges are non-monochromatic.
            for e in edges:
                colors = [assign[x] for x in e]
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False # Found a valid coloring
            return True # No valid coloring found
            
        # Try 0
        assign[v] = 0
        # Check if this causes any edge to be forced to fail?
        # If any edge becomes monochromatic with all vertices assigned, fail.
        # If any edge has 4 assigned 0 and 1 unassigned, that unassigned must be 1.
        # This is complex to implement generally.
        # Given n is small, simple backtracking might suffice for the few checks we do.
        # But to be safe, let's add a simple check:
        # If we set v to 0, is there an edge where all other 4 are 0 and the 5th is unassigned?
        # Then the 5th is forced to 1.
        
        # For simplicity and speed, let's just do standard backtracking.
        # With n=11, 2^11 = 2048 colorings. Very fast.
        # With n=26, 2^26 = 67 million. Too slow.
        # But we will likely use n=11 or 12.
        
        if backtrack(assign):
            return True
        assign[v] = 1
        if backtrack(assign):
            return True
        assign[v] = -1
        return False

    assign = [-1] * n
    return backtrack(assign)
=======
def is_property_b(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring.
    Returns True if Property B holds (no 2-coloring).
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Convert edges to tuples for faster hashing/comparison if needed, 
    # but we just iterate.
    
    # Precompute vertex indices for each edge for faster access
    edge_verts = [list(e) for e in edges]
    
    assign = [-1] * n
    
    def backtrack():
        # Check for conflicts and propagations
        # 1. Conflict: Edge is monochromatic (all 0 or all 1)
        # 2. Propagation: Edge has 4 same, 1 unassigned -> force the 5th
        
        # We iterate edges. If we find a conflict, return False (no valid coloring in this branch).
        # If we find a propagation, we set the variable.
        # We must be careful: if we set a variable, we might create new conflicts or propagations.
        # Simple approach: loop until no changes, then check for conflicts.
        
        changed = True
        while changed:
            changed = False
            for e in edge_verts:
                colors = [assign[v] for v in e]
                
                # Check conflict
                if -1 not in colors:
                    if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                        return False
                
                # Check propagation
                if colors.count(0) == 4 and -1 in colors:
                    for v in e:
                        if assign[v] == -1:
                            assign[v] = 1
                            changed = True
                elif colors.count(1) == 4 and -1 in colors:
                    for v in e:
                        if assign[v] == -1:
                            assign[v] = 0
                            changed = True
                            
        # Find next unassigned
        v = -1
        for i in range(n):
            if assign[i] == -1:
                v = i
                break
                
        if v == -1:
            # All assigned. 
            # We already checked for conflicts in the loop above? 
            # The loop above checks for conflicts only if -1 not in colors.
            # Since all are assigned, we just checked all edges. 
            # If no conflict was found, it's a valid coloring.
            return True
            
        # Try 0
        assign[v] = 0
        if backtrack():
            return True
        assign[v] = 1
        if backtrack():
            return True
        assign[v] = -1
        return False

    # We search for a valid 2-coloring.
    # If found, return False (Property B does NOT hold).
    # If not found, return True (Property B holds).
    return not backtrack()
>>>>>>> REPLACE

