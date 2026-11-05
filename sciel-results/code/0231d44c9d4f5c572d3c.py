"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def _check_unsat(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring using DPLL with unit propagation.
    """
    clauses = []
    for edge in edges:
        clauses.append([(v, 1) for v in edge])
        clauses.append([(v, -1) for v in edge])
    
    def dpll(clauses, assignment):
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                satisfied = False
                remaining = []
                for (var, sign) in clause:
                    if var in assignment:
                        val = assignment[var]
                        if (val and sign == 1) or (not val and sign == -1):
                            satisfied = True
                            break
                    else:
                        remaining.append((var, sign))
                if satisfied:
                    continue
                if not remaining:
                    return None
                if len(remaining) == 1:
                    var, sign = remaining[0]
                    if var in assignment:
                        val = assignment[var]
                        if (val and sign == 1) or (not val and sign == -1):
                            continue
                        else:
                            return None
                    assignment[var] = (sign == 1)
                    changed = True
                else:
                    new_clauses.append(remaining)
            clauses = new_clauses
        
        if not clauses:
            return assignment
        
        # Choose variable: pick from shortest clause, then most frequent
        # For simplicity and speed in small instances, pick var from first clause
        # Or better: pick var with highest frequency
        var_count = {}
        for clause in clauses:
            for (var, sign) in clause:
                var_count[var] = var_count.get(var, 0) + 1
        
        if not var_count:
            return assignment
            
        # Tie-breaker: random or index
        max_count = max(var_count.values())
        candidates = [v for v, c in var_count.items() if c == max_count]
        var = random.choice(candidates)
        
        # Try True
        assignment[var] = True
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        # Try False
        assignment[var] = False
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        return None
    
    result = dpll(clauses, {})
    return result is None

def _try_remove_edges(n, edges):
    """
    Try to remove edges from the hypergraph while maintaining non-2-colorability.
    Returns the minimal set of edges found.
    """
    current_edges = list(edges)
    random.shuffle(current_edges)
    
    # We will iterate through edges and try to remove them
    # If removing an edge makes it 2-colorable, we keep it.
    # We do this in multiple passes to handle dependencies.
    
    # To be efficient, we can try to remove edges in a random order repeatedly
    # until no more can be removed.
    
    removed = True
    while removed:
        removed = False
        # Shuffle to avoid bias
        random.shuffle(current_edges)
        for i in range(len(current_edges)):
            # Try removing current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            if not _check_unsat(n, test_edges):
                # It became 2-colorable, so we must keep this edge
                pass
            else:
                # It is still non-2-colorable, so we can remove it
                current_edges = test_edges
                removed = True
                # Restart loop with new list
                break
                
    return current_edges

def construct():
    best_result = None
    best_edges = float('inf')
    
    random.seed(42)
    
    # Search for n in 9..14
    for n in range(9, 15):
        all_possible = list(combinations(range(n), 5))
        
        # Try to start with a random subset
        for trial in range(50):
            # Start with a random number of edges, say 40-60
            num_edges = random.randint(40, 60)
            if num_edges > len(all_possible):
                num_edges = len(all_possible)
            
            selected = random.sample(all_possible, num_edges)
            edges = [list(e) for e in selected]
            
            # Check if non-2-colorable
            if _check_unsat(n, edges):
                # Try to minimize
                minimized_edges = _try_remove_edges(n, edges)
                
                if len(minimized_edges) < best_edges:
                    best_edges = len(minimized_edges)
                    best_result = {'n': n, 'edges': minimized_edges}
                    
    if best_result is not None:
        return best_result
        
    # Fallback: complete graph on 9 vertices (126 edges)
    # Or a known 51-edge construction if we had it hardcoded.
    # For now, return a dense one that is definitely non-2-colorable.
    n = 9
    return {'n': n, 'edges': [list(e) for e in combinations(range(n), 5)]}

# EVOLVE-BLOCK-END
