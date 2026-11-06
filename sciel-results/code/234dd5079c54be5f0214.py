import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses a DPLL-style solver with unit propagation.
    """
    if not edges:
        return True

    # Clauses representation:
    # Each edge e gives two clauses:
    # 1. OR of all v in e (at least one is 1)
    # 2. OR of all -v in e (at least one is 0)
    # We represent literals as integers: v+1 for positive, -(v+1) for negative.
    
    clauses = []
    for e in edges:
        pos_clause = tuple(v + 1 for v in e)
        neg_clause = tuple(-(v + 1) for v in e)
        clauses.append(set(pos_clause))
        clauses.append(set(neg_clause))
        
    # Remove duplicate clauses
    clauses = list(set(clauses))
    
    # If any clause is empty, it's unsat
    for c in clauses:
        if len(c) == 0:
            return False
            
    assignment = {} # var -> 0 or 1
    
    def propagate(clauses, assignment):
        """
        Apply unit propagation.
        Returns False if conflict, True otherwise.
        Modifies clauses and assignment in place (clauses list is replaced).
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                # Check for satisfied or falsified literals
                unsat = False
                satisfied = False
                remaining = set()
                for lit in c:
                    var = abs(lit) - 1
                    if var in assignment:
                        val = assignment[var]
                        # lit is positive if lit > 0
                        # If val is 1, positive lit is true, negative is false
                        # If val is 0, positive lit is false, negative is true
                        if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                            satisfied = True
                            break
                        elif (lit > 0 and val == 0) or (lit < 0 and val == 1):
                            unsat = True
                            break
                    else:
                        remaining.add(lit)
                
                if satisfied:
                    continue # Clause satisfied, discard
                if unsat:
                    return False # Conflict
                if len(remaining) == 0:
                    return False # Empty clause, conflict
                
                if len(remaining) == 1:
                    # Unit clause
                    lit = next(iter(remaining))
                    var = abs(lit) - 1
                    val = 1 if lit > 0 else 0
                    if var in assignment:
                        if assignment[var] != val:
                            return False
                    else:
                        assignment[var] = val
                        changed = True
                        # Discard this clause as it is now satisfied
                        continue
                
                new_clauses.append(remaining)
            clauses = new_clauses
        return True

    def dpll(clauses, assignment):
        if not clauses:
            return True # Satisfiable
        
        # Check for empty clauses (should be handled by propagate, but safe)
        for c in clauses:
            if len(c) == 0:
                return False
        
        # Unit propagation
        if not propagate(clauses, assignment):
            return False
            
        if not clauses:
            return True
            
        # Heuristic: pick a variable from the shortest clause
        # Shortest clause is most constrained
        min_len = min(len(c) for c in clauses)
        var_to_pick = -1
        for c in clauses:
            if len(c) == min_len:
                lit = next(iter(c))
                var_to_pick = abs(lit) - 1
                break
                
        if var_to_pick == -1:
            return True # All assigned?
            
        # Try 1
        assignment[var_to_pick] = 1
        if dpll(clauses[:], assignment):
            return True
        del assignment[var_to_pick]
        
        # Try 0
        assignment[var_to_pick] = 0
        if dpll(clauses[:], assignment):
            return True
        del assignment[var_to_pick]
        
        return False

    return dpll(clauses, assignment)

def construct():
    import random
    from itertools import combinations
    
    best_edges = None
    best_len = float('inf')
    best_n = 0
    
    # Try different n values. 
    random.seed(42)
    
    for n in range(12, 19):
        all_edges = list(combinations(range(n), 5))
        total_edges = len(all_edges)
        
        # Strategy: Start with a dense random subgraph
        found = False
        current_edges = None
        
        for attempt in range(20):
            target_size = min(total_edges, 60 + random.randint(0, 30))
            edges = random.sample(all_edges, target_size)
            edges = [list(e) for e in edges]
            
            if not is_2_colorable(n, edges):
                found = True
                current_edges = edges
                break
        
        if not found:
            continue
            
        # Minimize the core
        random.shuffle(current_edges)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(current_edges) - 1, -1, -1):
                e_to_remove = current_edges[i]
                temp_edges = current_edges[:i] + current_edges[i+1:]
                
                if not is_2_colorable(n, temp_edges):
                    current_edges.pop(i)
                    changed = True
                    
        if len(current_edges) < best_len:
            best_len = len(current_edges)
            best_edges = current_edges
            best_n = n
            
    if best_edges is None:
        # Fallback: Complete hypergraph on 9 vertices
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}
