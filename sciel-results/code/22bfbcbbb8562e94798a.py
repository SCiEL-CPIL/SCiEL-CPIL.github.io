import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try different n values and random seeds
    # Known constructions are often around n=15-20
    n_candidates = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Shuffle to avoid biasing early time towards small n if they fail
    random.shuffle(n_candidates)
    
    for n in n_candidates:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each n
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            edges = []
            edge_set = set()
            
            # Greedy construction
            # We don't need to pre-generate all edges, just check potential ones
            # But for n<=20, total 5-edges is C(20,5)=15504, which is small.
            all_potential_edges = [tuple(sorted(e)) for e in combinations(range(n), 5)]
            
            # Initialize with no edges
            # We need a way to check 2-colorability
            # If it's 2-colorable, find a coloring, pick a monochromatic edge from it, add to E.
            
            while True:
                if time.time() > deadline:
                    break
                
                # Check if current E is Property B (not 2-colorable)
                is_colorable, coloring = _check_2_colorable(n, edges)
                
                if not is_colorable:
                    # Found a valid hypergraph
                    if len(edges) < best_count:
                        best_count = len(edges)
                        best_edges = [list(e) for e in edges]
                        best_n = n
                    break
                else:
                    # It is colorable. Get a valid coloring.
                    # Find all 5-edges that are monochromatic in this coloring.
                    mono_edges = []
                    # Optimization: Only check edges not in E? 
                    # Actually, we want to ADD an edge. 
                    # Any 5-subset that is mono in the current coloring is a candidate.
                    # If we add it, this specific coloring becomes invalid.
                    for e in all_potential_edges:
                        if e in edge_set:
                            continue
                        # Check if e is monochromatic in 'coloring'
                        c0 = coloring[e[0]]
                        if all(coloring[v] == c0 for v in e):
                            mono_edges.append(e)
                    
                    if not mono_edges:
                        # This shouldn't happen if is_colorable is true?
                        # If a coloring is valid, no edge in E is mono.
                        # But there might be no mono edges among potential edges?
                        # If n is small, maybe all potential edges are already in E?
                        # If all potential edges are in E and it's still colorable, 
                        # then the complete hypergraph is colorable? No, complete hypergraph on n>=5 is not 2-colorable?
                        # Wait. Complete 5-uniform hypergraph on 9 vertices is NOT 2-colorable.
                        # If we have all edges and it's colorable, it's a contradiction.
                        # So if we run out of candidates, something is wrong.
                        # However, if we have a valid coloring, and we pick a random edge from ALL potential edges 
                        # that is mono, we guarantee progress.
                        # If mono_edges is empty, it means NO potential 5-edge is monochromatic in this coloring.
                        # This implies the coloring is "balanced" enough that no 5-set is mono.
                        # In that case, we can just pick a random potential edge not in E.
                        remaining = [e for e in all_potential_edges if e not in edge_set]
                        if not remaining:
                            break
                        e_add = random.choice(remaining)
                    else:
                        e_add = random.choice(mono_edges)
                    
                    edges.append(e_add)
                    edge_set.add(e_add)
                    
                    # If we exceed a reasonable number of edges, abort this restart
                    if len(edges) > 60:
                        break

    if best_edges is None:
        # Fallback: Return the seed program's result (complete graph on 9)
        # This is valid but not minimal.
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}

def _check_2_colorable(n, edges):
    """
    Checks if the hypergraph (n, edges) is 2-colorable.
    Returns (True, coloring) if colorable, (False, None) if not.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True, [0] * n
    
    # Map vertices to indices 0..n-1
    # State: list of -1 (unknown), 0, 1
    assignment = [-1] * n
    
    # Pre-process edges to tuples for faster lookup
    edge_tuples = [tuple(e) for e in edges]
    
    def propagate(assignment):
        """
        Applies unit propagation.
        Returns False if conflict, True if no conflict (and modifies assignment).
        """
        changed = True
        while changed:
            changed = False
            for e in edge_tuples:
                # Count colors in this edge
                c0 = 0
                c1 = 0
                unknown = 0
                for v in e:
                    a = assignment[v]
                    if a == 0:
                        c0 += 1
                    elif a == 1:
                        c1 += 1
                    else:
                        unknown += 1
                
                # Check for conflict: if all 5 are same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit Propagation:
                # If 4 are 0, the 5th must be 1
                if c0 == 4 and unknown == 1:
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            changed = True
                            break
                            
                # If 4 are 1, the 5th must be 0
                elif c1 == 4 and unknown == 1:
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            changed = True
                            break
                            
                # Note: If c0=3, c1=1, unknown=1, no forcing.
                # If c0=3, c1=0, unknown=2, no forcing.
                
        return True

    def backtrack(assignment):
        # Find first unassigned vertex
        idx = -1
        for i in range(n):
            if assignment[i] == -1:
                idx = i
                break
                
        if idx == -1:
            # All assigned
            # Final check for conflicts (though propagation should have caught most)
            for e in edge_tuples:
                c0 = sum(1 for v in e if assignment[v] == 0)
                c1 = sum(1 for v in e if assignment[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
            
        # Try 0
        old_val = assignment[idx]
        assignment[idx] = 0
        if propagate(assignment.copy()):
            if backtrack(assignment.copy()):
                return True
        assignment[idx] = old_val
        
        # Try 1
        assignment[idx] = 1
        if propagate(assignment.copy()):
            if backtrack(assignment.copy()):
                return True
        assignment[idx] = old_val
        
        return False

    # Initial propagation
    if not propagate(assignment.copy()):
        return False, None
        
    # Check if any unassigned
    if -1 in assignment:
        if backtrack(assignment.copy()):
            # Need to retrieve the final assignment. 
            # The backtrack function above returns bool but doesn't return the assignment.
            # Let's modify backtrack to return the assignment or use a global/outer var.
            # For simplicity and speed in this context, we'll just return True/False and 
            # if True, we need the coloring.
            # Let's re-run a simple check to get the coloring if it exists?
            # Or better, modify backtrack to store the solution.
            
            # Refactored backtrack to store solution
            solution = None
            def backtrack_sol(assignment):
                nonlocal solution
                idx = -1
                for i in range(n):
                    if assignment[i] == -1:
                        idx = i
                        break
                if idx == -1:
                    for e in edge_tuples:
                        c0 = sum(1 for v in e if assignment[v] == 0)
                        if c0 == 5: return False
                        c1 = sum(1 for v in e if assignment[v] == 1)
                        if c1 == 5: return False
                    solution = assignment[:]
                    return True
                    
                assignment[idx] = 0
                if propagate(assignment.copy()):
                    if backtrack_sol(assignment.copy()):
                        return True
                assignment[idx] = 1
                if propagate(assignment.copy()):
                    if backtrack_sol(assignment.copy()):
                        return True
                assignment[idx] = -1
                return False

            if backtrack_sol(assignment.copy()):
                return True, solution
            else:
                return False, None
        else:
            return False, None
    else:
        # All assigned, check conflicts
        for e in edge_tuples:
            c0 = sum(1 for v in e if assignment[v] == 0)
            if c0 == 5:
                return False, None
            c1 = sum(1 for v in e if assignment[v] == 1)
            if c1 == 5:
                return False, None
        return True, assignment

# Helper to speed up propagation by avoiding copy overhead in tight loops if possible, 
# but for n<=20, copy is fine.
# The above _check_2_colorable is a bit heavy due to copies. 
# Let's optimize _check_2_colorable slightly for the 150s budget.

def _check_2_colorable_optimized(n, edges):
    if not edges:
        return True, [0] * n
        
    edge_tuples = [tuple(e) for e in edges]
    assignment = [-1] * n
    
    def propagate_in_place(assignment):
        changed = True
        while changed:
            changed = False
            for e in edge_tuples:
                c0 = 0
                c1 = 0
                unknown_idx = -1
                unknown_count = 0
                for v in e:
                    a = assignment[v]
                    if a == 0:
                        c0 += 1
                    elif a == 1:
                        c1 += 1
                    else:
                        unknown_count += 1
                        unknown_idx = v
                        
                if c0 == 5 or c1 == 5:
                    return False
                    
                if c0 == 4 and unknown_count == 1:
                    assignment[unknown_idx] = 1
                    changed = True
                elif c1 == 4 and unknown_count == 1:
                    assignment[unknown_idx] = 0
                    changed = True
        return True

    def backtrack(assignment):
        idx = -1
        min_unknowns = float('inf')
        # Heuristic: pick vertex in edge with most constraints? 
        # Simple: first unassigned
        for i in range(n):
            if assignment[i] == -1:
                idx = i
                break
                
        if idx == -1:
            for e in edge_tuples:
                if all(assignment[v] == 0 for v in e) or all(assignment[v] == 1 for v in e):
                    return False
            return True
            
        # Try 0
        assignment[idx] = 0
        if propagate_in_place(assignment):
            if backtrack(assignment):
                return True
        assignment[idx] = -1
        
        # Try 1
        assignment[idx] = 1
        if propagate_in_place(assignment):
            if backtrack(assignment):
                return True
        assignment[idx] = -1
        
        return False

    if not propagate_in_place(assignment):
        return False, None
        
    if -1 in assignment:
        if backtrack(assignment):
            return True, assignment
        else:
            return False, None
    else:
        for e in edge_tuples:
            if all(assignment[v] == 0 for v in e) or all(assignment[v] == 1 for v in e):
                return False, None
        return True, assignment

# Replace the original check with the optimized one in construct
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    n_candidates = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    random.shuffle(n_candidates)
    
    for n in n_candidates:
        if time.time() > deadline:
            break
            
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            edges = []
            edge_set = set()
            
            all_potential_edges = [tuple(sorted(e)) for e in combinations(range(n), 5)]
            
            while True:
                if time.time() > deadline:
                    break
                
                is_colorable, coloring = _check_2_colorable_optimized(n, edges)
                
                if not is_colorable:
                    if len(edges) < best_count:
                        best_count = len(edges)
                        best_edges = [list(e) for e in edges]
                        best_n = n
                    break
                else:
                    mono_edges = []
                    for e in all_potential_edges:
                        if e in edge_set:
                            continue
                        c0 = coloring[e[0]]
                        if all(coloring[v] == c0 for v in e):
                            mono_edges.append(e)
                    
                    if not mono_edges:
                        remaining = [e for e in all_potential_edges if e not in edge_set]
                        if not remaining:
                            break
                        e_add = random.choice(remaining)
                    else:
                        e_add = random.choice(mono_edges)
                    
                    edges.append(e_add)
                    edge_set.add(e_add)
                    
                    if len(edges) > 60:
                        break

    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}

