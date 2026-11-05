"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations
import time

def _check_unsat(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring using DPLL with unit propagation.
    For each edge, we add two clauses: not(all 0) and not(all 1).
    We check if the resulting CNF is UNSAT.
    """
    # Build clauses as lists of (var, sign) where sign=1 means var, sign=-1 means not var
    clauses = []
    for edge in edges:
        # Clause: at least one vertex is color 1 -> (v0 or v1 or v2 or v3 or v4)
        clauses.append([(v, 1) for v in edge])
        # Clause: at least one vertex is color 0 -> (not v0 or not v1 or not v2 or not v3 or not v4)
        clauses.append([(v, -1) for v in edge])
    
    def dpll(clauses, assignment):
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Simplify clause
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
                    return None  # UNSAT
                if len(remaining) == 1:
                    var, sign = remaining[0]
                    if var in assignment:
                        val = assignment[var]
                        if (val and sign == 1) or (not val and sign == -1):
                            continue
                        else:
                            return None  # UNSAT
                    assignment[var] = (sign == 1)
                    changed = True
                else:
                    new_clauses.append(remaining)
            clauses = new_clauses
        
        if not clauses:
            return assignment  # SAT
        
        # Choose a variable to branch on using failed literal probing or simply most frequent
        var_count = {}
        for clause in clauses:
            for (var, sign) in clause:
                var_count[var] = var_count.get(var, 0) + 1
        if not var_count:
            return assignment
        
        var = max(var_count, key=lambda v: var_count[v])
        
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
        
        return None  # UNSAT
    
    result = dpll(clauses, {})
    return result is None  # True if UNSAT (no 2-coloring)


def _generate_weighted_hypergraph(n, num_edges, density_bias=1.0):
    """
    Generate a random 5-uniform hypergraph with weighted edge selection.
    density_bias > 1 favors edges with more overlap with existing edges.
    """
    if num_edges <= 0:
        return {'n': n, 'edges': []}
    
    all_possible = list(combinations(range(n), 5))
    
    # Start with a random edge
    edges = [list(all_possible[random.randint(0, len(all_possible)-1)])]
    
    while len(edges) < num_edges:
        # Weight candidates by how many of their 4-subsets appear in existing edges
        weights = []
        for i, edge in enumerate(all_possible):
            if i in [tuple(e) for e in edges]:
                weights.append(0)
                continue
            weight = 1.0
            # Count 4-subsets that appear in existing edges
            for j in range(5):
                subset = tuple(sorted(edge[:j] + edge[j+1:]))
                for existing_edge in edges:
                    existing_subsets = [tuple(sorted(existing_edge[:k] + existing_edge[k+1:])) for k in range(5)]
                    if subset in existing_subsets:
                        weight += density_bias
                        break
            weights.append(weight)
        
        # Select an edge with probability proportional to weight
        total_weight = sum(weights)
        if total_weight == 0:
            # All edges already selected, pick randomly from remaining
            remaining = [i for i in range(len(all_possible)) if i not in [tuple(e) for e in edges]]
            if not remaining:
                break
            idx = remaining[random.randint(0, len(remaining)-1)]
        else:
            r = random.random() * total_weight
            cumulative = 0
            idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    idx = i
                    break
        
        edges.append(list(all_possible[idx]))
    
    return {'n': n, 'edges': edges}


def construct():
    """
    Find a minimal non-2-colorable 5-uniform hypergraph.
    """
    best_result = None
    best_edges = float('inf')
    
    start_time = time.time()
    time_limit = 115  # Leave 5 seconds buffer
    
    # Strategy: Try to find hypergraphs with fewer edges than 51
    # Focus on n=10-14 vertices with 29-50 edges
    
    random.seed(42)
    
    # First, try to find a solution with fewer edges
    # Try configurations in order of increasing edge count
    configs = []
    for num_edges in range(29, 52):
        for n in range(10, 15):
            configs.append((n, num_edges))
    
    # Shuffle to try different n values at each edge count
    for num_edges in range(29, 52):
        ns = list(range(10, 15))
        random.shuffle(ns)
        for n in ns:
            configs.append((n, num_edges))
    
    # Also include some higher edge counts as fallback
    for num_edges in range(52, 60):
        for n in range(10, 15):
            configs.append((n, num_edges))
    
    for n, num_edges in configs:
        if time.time() - start_time > time_limit:
            break
        if num_edges >= best_edges:
            continue
        
        # Try multiple random instances with different density biases
        for bias in [0.5, 1.0, 2.0, 3.0]:
            if time.time() - start_time > time_limit:
                break
            
            for trial in range(5):
                if time.time() - start_time > time_limit:
                    break
                
                hg = _generate_weighted_hypergraph(n, num_edges, density_bias=bias)
                if len(hg['edges']) != num_edges:
                    continue
                    
                if _check_unsat(hg['n'], hg['edges']):
                    if num_edges < best_edges:
                        best_edges = num_edges
                        best_result = hg
                        break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a known valid hypergraph
    # Use complete 5-graph on 9 vertices (126 edges) as guaranteed fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
