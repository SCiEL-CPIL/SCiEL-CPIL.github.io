import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Preprocess: map edges to indices for faster lookup? 
    # Or just use lists.
    
    # We'll use a recursive function.
    # State: color assignment array (-1 for unassigned, 0 or 1 for assigned)
    # To make it fast, we can use a list of constraints.
    
    # Let's use a simple DPLL on the SAT formulation.
    # Variables: 0..n-1
    # Clauses: 
    # For each edge e: 
    #   clause 1: x_v1 or x_v2 or x_v3 or x_v4 or x_v5
    #   clause 2: not x_v1 or not x_v2 or not x_v3 or not x_v4 or not x_v5
    
    # We'll implement a simple solver.
    
    # Optimization: Remove duplicate edges? The task says distinct edges.
    
    # Let's represent clauses as sets of (var, sign).
    # sign: 1 for positive, -1 for negative.
    
    clauses = []
    for e in edges:
        # Positive clause
        pos_clause = [(v, 1) for v in e]
        # Negative clause
        neg_clause = [(v, -1) for v in e]
        clauses.append(pos_clause)
        clauses.append(neg_clause)
        
    # Initial assignment
    assignment = [-1] * n
    
    def propagate(clauses, assignment):
        """
        Applies unit propagation.
        Returns False if conflict, True if no conflict (and updates assignment/clauses).
        """
        changed = True
        while changed:
            changed = False
            for i, clause in enumerate(clauses):
                if not clause:
                    # Empty clause: conflict
                    return False
                if len(clause) == 1:
                    var, sign = clause[0]
                    # Check if already assigned
                    if assignment[var] == -1:
                        assignment[var] = sign
                        changed = True
                    elif assignment[var] != sign:
                        return False
                    # Remove this literal from all clauses? 
                    # Actually, standard DPLL simplifies clauses.
                    # To keep it simple and correct, we'll just rely on the check in the main loop 
                    # or re-evaluate clause lengths. 
                    # Re-evaluating all clauses every time is O(C*L). 
                    # For small instances, this is fine.
                    # Let's simplify: remove the satisfied literal from the clause, 
                    # and if the clause becomes empty, conflict.
                    # If the clause becomes unit, handle in next iteration.
                    # We need to modify clauses in place.
                    
                    # Wait, if we modify clauses, we need to be careful.
                    # Let's just re-scan.
                    
                    # Actually, a cleaner way for small n:
                    # Just check all clauses in the main backtracking step.
                    # But unit propagation is key for speed.
                    
                    # Let's do a proper simplification step.
                    # Remove the literal (var, sign) from all clauses.
                    # If a clause becomes empty, return False.
                    # If a clause becomes unit, it will be handled in the next iteration of the while loop.
                    
                    # To avoid modifying the original list structure too much, 
                    # we can just mark it and filter.
                    pass
                    
        return True

    # The above propagation is a bit messy to implement correctly with in-place modification 
    # without deep copies. 
    # Given n <= 26 and edges <= 200, a simpler backtracking without explicit 
    # clause simplification might work if we use good heuristics, 
    # but unit propagation is very powerful.
    
    # Let's try a different approach: 
    # Use a bitmask for assigned variables and values.
    # But unit propagation requires checking all edges.
    
    # Let's stick to a simple recursive backtracking that checks constraints 
    # and uses a "most constrained variable" heuristic.
    
    def backtrack(assigned_count):
        if assigned_count == n:
            return True
            
        # Find unassigned variable
        # Heuristic: pick the variable that appears in the most "critical" edges
        # Critical edge: an edge where all but 1 vertices are assigned.
        # If such an edge exists, the unassigned vertex is forced.
        
        forced_var = -1
        forced_val = -1
        for e in edges:
            unassigned_in_e = [v for v in e if assignment[v] == -1]
            if len(unassigned_in_e) == 0:
                # All assigned. Check if monochromatic
                colors = [assignment[v] for v in e]
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False
            elif len(unassigned_in_e) == 1:
                # One unassigned. It is forced.
                v = unassigned_in_e[0]
                colors = [assignment[u] for u in e if u != v]
                if all(c == 0 for c in colors):
                    # Must be 1
                    forced_var = v
                    forced_val = 1
                elif all(c == 1 for c in colors):
                    # Must be 0
                    forced_var = v
                    forced_val = 0
                else:
                    # Mixed, so v can be 0 or 1, no immediate force from this edge
                    pass
                    
        if forced_var != -1:
            assignment[forced_var] = forced_val
            res = backtrack(assigned_count + 1)
            assignment[forced_var] = -1
            return res
            
        # No forced variable. Pick an unassigned variable.
        # Heuristic: pick one with highest degree in remaining graph? 
        # Or just the first unassigned.
        var_to_assign = -1
        for v in range(n):
            if assignment[v] == -1:
                var_to_assign = v
                break
                
        if var_to_assign == -1:
            return True # Should not happen if assigned_count < n
            
        # Try 0
        assignment[var_to_assign] = 0
        # Check if any edge becomes monochromatic 0
        valid = True
        for e in edges:
            if var_to_assign in e:
                colors = [assignment[v] for v in e]
                if all(c == 0 for c in colors):
                    valid = False
                    break
                # Also check if any edge is "almost" monochromatic? 
                # The forced check in the next recursion will handle it.
                # But we can prune earlier: if an edge has all 1s and this var is 0, it's fine.
                # If an edge has all 0s and this var is 0, it's invalid.
                # We already check that.
                
        if valid:
            if backtrack(assigned_count + 1):
                return True
                
        # Try 1
        assignment[var_to_assign] = 1
        valid = True
        for e in edges:
            if var_to_assign in e:
                colors = [assignment[v] for v in e]
                if all(c == 1 for c in colors):
                    valid = False
                    break
                    
        if valid:
            if backtrack(assigned_count + 1):
                return True
                
        assignment[var_to_assign] = -1
        return False
        
    return backtrack(0)

def construct():
    import random
    from itertools import combinations
    
    # Set a random seed for reproducibility if needed, but random is fine.
    random.seed(42)
    
    best_edges = None
    best_n = 0
    
    # Try different n values
    for n in range(10, 16):
        # Generate a random hypergraph
        all_edges = list(combinations(range(n), 5))
        
        # Start with a random subset
        # We want it to be non-2-colorable.
        # Dense graphs are more likely to be non-2-colorable.
        # Let's start with all edges and remove?
        # Or start with a random set of size ~100.
        
        # Let's try to find a non-2-colorable instance first.
        found = False
        for attempt in range(20):
            # Random sample
            sample_size = min(len(all_edges), 100 + random.randint(0, 50))
            edges = random.sample(all_edges, sample_size)
            edges = [list(e) for e in edges]
            
            if not is_2_colorable(n, edges):
                # We have a non-2-colorable hypergraph. Now minimize.
                found = True
                break
        
        if not found:
            continue
            
        # Minimize edges
        # Sort edges? Maybe remove edges that are "redundant".
        # Iterate through edges and try to remove them.
        # To make it faster, we can shuffle the order.
        
        current_edges = edges[:]
        
        # Repeat passes until no change
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(current_edges)
            for i in range(len(current_edges) - 1, -1, -1):
                e_to_remove = current_edges[i]
                # Try removing it
                temp_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if still non-2-colorable
                if not is_2_colorable(n, temp_edges):
                    # It is still non-2-colorable, so we can remove the edge
                    current_edges.pop(i)
                    changed = True
                    # Restart the pass? Or continue?
                    # Continuing is fine, but the list size changed.
                    # Since we are iterating backwards, it's safe.
                    # But we should probably restart to be safe and find more removals.
                    # Actually, if we remove one, other edges might become redundant too.
                    # Breaking and restarting is safer.
                    break
        
        if len(current_edges) < (best_n if best_edges is None else len(best_edges)):
            best_edges = current_edges
            best_n = n
            
    if best_edges is None:
        # Fallback to seed solution
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
