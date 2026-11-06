import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Build adjacency: for each variable, which edges contain it
    var_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            var_edges[v].append(i)
    
    assignment = [-1] * n
    edge_status = [0] * len(edges)  # 0: active, 1: satisfied (has both colors), 2: violated (monochromatic)
    
    def count_unassigned(e):
        count = 0
        for v in e:
            if assignment[v] == -1:
                count += 1
        return count
    
    def check_edge(e):
        """Check if edge is satisfied, violated, or still pending"""
        has_0 = False
        has_1 = False
        unassigned = 0
        for v in e:
            if assignment[v] == 0:
                has_0 = True
            elif assignment[v] == 1:
                has_1 = True
            else:
                unassigned += 1
        if has_0 and has_1:
            return 1  # satisfied
        if unassigned == 0:
            return 2  # violated (monochromatic)
        return 0  # pending
    
    def propagate():
        """
        Apply unit propagation. Returns False if conflict detected.
        Modifies assignment in place.
        """
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                if edge_status[i] == 1:  # already satisfied
                    continue
                if edge_status[i] == 2:  # already violated
                    return False
                
                status = check_edge(e)
                if status == 2:
                    return False
                if status == 1:
                    edge_status[i] = 1
                    changed = True
                    continue
                
                # Edge is pending. Check for forced assignments.
                # If all but one are 0, the last must be 1.
                # If all but one are 1, the last must be 0.
                zeros = 0
                ones = 0
                unassigned_vars = []
                for v in e:
                    if assignment[v] == 0:
                        zeros += 1
                    elif assignment[v] == 1:
                        ones += 1
                    else:
                        unassigned_vars.append(v)
                
                if zeros == 4 and len(unassigned_vars) == 1:
                    # All assigned are 0, so the unassigned must be 1
                    v = unassigned_vars[0]
                    if assignment[v] != -1:
                        if assignment[v] != 1:
                            return False
                    else:
                        assignment[v] = 1
                        changed = True
                elif ones == 4 and len(unassigned_vars) == 1:
                    # All assigned are 1, so the unassigned must be 0
                    v = unassigned_vars[0]
                    if assignment[v] != -1:
                        if assignment[v] != 0:
                            return False
                    else:
                        assignment[v] = 0
                        changed = True
                elif zeros == 5:
                    return False
                elif ones == 5:
                    return False
        
        return True
    
    def backtrack():
        if propagate():
            # Find unassigned variable
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                return True
            
            # Heuristic: pick variable with highest degree in pending edges
            best_var = None
            best_score = -1
            for v in unassigned:
                score = 0
                for ei in var_edges[v]:
                    if edge_status[ei] == 0:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_var = v
            
            # Try 0
            assignment[best_var] = 0
            if backtrack():
                return True
            assignment[best_var] = -1
            
            # Try 1
            assignment[best_var] = 1
            if backtrack():
                return True
            assignment[best_var] = -1
            
            return False
        else:
            return False
    
    return backtrack()

def construct():
    random.seed(42)
    
    best_edges = None
    best_n = 0
    
    # Try different n values and random seeds
    for n in range(10, 17):
        all_edges = list(combinations(range(n), 5))
        
        for attempt in range(30):
            # Generate a random hypergraph with enough edges to likely be non-colorable
            sample_size = min(len(all_edges), 80 + random.randint(0, 40))
            edges = [list(e) for e in random.sample(all_edges, sample_size)]
            
            # Check if non-2-colorable
            if is_2_colorable(n, edges):
                continue
            
            # Extract minimal unsatisfiable core
            current_edges = edges[:]
            changed = True
            while changed:
                changed = False
                # Shuffle for diversity
                random.shuffle(current_edges)
                for i in range(len(current_edges)):
                    e_to_remove = current_edges[i]
                    temp_edges = current_edges[:i] + current_edges[i+1:]
                    
                    if not is_2_colorable(n, temp_edges):
                        # Still non-colorable, remove the edge
                        current_edges.pop(i)
                        changed = True
                        break  # Restart pass
            
            if best_edges is None or len(current_edges) < len(best_edges):
                best_edges = current_edges
                best_n = n
        
        # Early stop if we found a good solution
        if best_edges is not None and len(best_edges) <= 55:
            break
    
    if best_edges is None:
        # Fallback: use a known construction or dense graph
        n = 12
        all_edges = list(combinations(range(n), 5))
        edges = [list(e) for e in all_edges[:100]]
        # Verify
        if is_2_colorable(n, edges):
            edges = [list(e) for e in all_edges]
        best_edges = edges
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}
