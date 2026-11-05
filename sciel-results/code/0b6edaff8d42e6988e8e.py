"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
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
        
        var_count = {}
        for clause in clauses:
            for (var, sign) in clause:
                var_count[var] = var_count.get(var, 0) + 1
        if not var_count:
            return assignment
        
        var = max(var_count, key=lambda v: var_count[v])
        
        assignment[var] = True
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        assignment[var] = False
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        return None
    
    result = dpll(clauses, {})
    return result is None

def _generate_random_hypergraph(n, num_edges):
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    edges = [list(e) for e in all_edges[:num_edges]]
    return {'n': n, 'edges': edges}

def construct():
    random.seed(42)
    start_time = time.time()
    time_limit = 110.0
    
    best_result = None
    best_edges = float('inf')
    
    # We search for the smallest number of edges
    # Try n from 9 to 14
    # For each n, try M from 30 upwards
    # Stop if time limit is reached
    
    for n in range(9, 15):
        max_possible_edges = len(list(combinations(range(n), 5)))
        # Start M from a reasonable lower bound
        # For n=9, min edges to be non-2-colorable is > 28.
        # We'll start from 30.
        for M in range(30, min(max_possible_edges + 1, 127)):
            if time.time() - start_time > time_limit:
                break
            
            # Try several random instances for this M
            # More trials for smaller M to increase chance of finding a valid one
            trials = 200 if M < 50 else 50
            
            for _ in range(trials):
                if time.time() - start_time > time_limit:
                    break
                
                hg = _generate_random_hypergraph(n, M)
                if _check_unsat(hg['n'], hg['edges']):
                    if M < best_edges:
                        best_edges = M
                        best_result = hg
                    # If we found one, we can try to find a smaller one
                    # But since we iterate M upwards, this is the smallest M for this n
                    # We should break the M loop for this n?
                    # No, maybe a smaller M works for a larger n?
                    # Unlikely, but possible.
                    # However, if we found one at M=40 for n=10, 
                    # it's unlikely n=11 has a valid one at M=30.
                    # So we can break the M loop for this n and move to next n?
                    # Or just break all and return?
                    # To be safe, we continue searching for smaller M in other n?
                    # Actually, if we found M=40, we want to see if M=39 exists.
                    # But we are iterating M upwards. So we already passed M=39.
                    # So if we didn't find it then, we probably won't now.
                    # So we can break the M loop for this n.
                    break
            
            # If we found a valid hypergraph for this n and M, 
            # we can stop searching for larger M for this n.
            if best_result is not None and best_edges == M:
                # We found the smallest M for this n (within our trials)
                # Move to next n to see if a smaller M exists there
                continue 
            else:
                # No valid hypergraph found for this M
                # Continue to next M
                pass
        
        if time.time() - start_time > time_limit:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Complete 5-graph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
