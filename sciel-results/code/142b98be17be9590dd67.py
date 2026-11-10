"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""

import itertools
import random
import time

def construct():
    """
    Return a 5-uniform hypergraph on n vertices with no proper 2-colouring,
    minimizing the number of edges.
    
    Uses a randomized search strategy:
    1. Generate random hypergraphs.
    2. Check for 2-colorability using backtracking with unit propagation.
    3. If not 2-colorable, minimize edge count by greedy removal.
    4. If 2-colorable, add edges until not 2-colorable, then minimize.
    """
    
    def is_2_colorable(n, edges):
        """
        Return True if the hypergraph is 2-colorable (exists coloring where no edge is monochromatic).
        Uses backtracking with unit propagation.
        """
        m = len(edges)
        if m == 0:
            return True
        
        # Precompute
        edge_vertices = [set(e) for e in edges]
        
        # assignment: -1 unassigned, 0 or 1
        assignment = [-1] * n
        
        # For propagation, we need to know which edges are "critical"
        # An edge is critical if it has 4 assigned vertices of the same color.
        # Then the 5th must be the opposite color.
        
        def propagate():
            """
            Apply unit propagation.
            Returns False if a contradiction is found.
            Modifies assignment in place.
            """
            changed = True
            while changed:
                changed = False
                for e in edge_vertices:
                    vals = [assignment[v] for v in e]
                    # Count colors
                    c0 = sum(1 for v in vals if v == 0)
                    c1 = sum(1 for v in vals if v == 1)
                    unassigned = [v for v in e if assignment[v] == -1]
                    
                    if not unassigned:
                        # All assigned
                        if c0 == 0 or c1 == 0:
                            # Monochromatic -> Violation of "mixed" constraint
                            # Wait, the constraint is "no edge is monochromatic".
                            # So if an edge IS monochromatic, this coloring is INVALID for this edge.
                            # In the search for a valid coloring, if we reach a state where an edge is monochromatic, 
                            # this branch is a dead end.
                            return False
                    
                    # Check for unit propagation
                    # If 4 are 0, the remaining 1 MUST be 1.
                    if c0 == 4 and len(unassigned) == 1:
                        v = unassigned[0]
                        if assignment[v] == -1:
                            assignment[v] = 1
                            changed = True
                        elif assignment[v] == 0:
                            return False # Contradiction: forced to 1 but is 0
                            
                    # If 4 are 1, the remaining 1 MUST be 0.
                    elif c1 == 4 and len(unassigned) == 1:
                        v = unassigned[0]
                        if assignment[v] == -1:
                            assignment[v] = 0
                            changed = True
                        elif assignment[v] == 1:
                            return False # Contradiction: forced to 0 but is 1
                            
                    # If 3 are 0 and 1 is 1, the remaining 1 can be 0 or 1?
                    # If remaining is 0: 4 zeros, 1 one -> Mixed. OK.
                    # If remaining is 1: 3 zeros, 2 ones -> Mixed. OK.
                    # So no constraint.
                    
                    # If 0 are 0 and 0 are 1 (all unassigned), no constraint.
                    
                    # What if c0=5? Handled by "All assigned" check above? 
                    # No, "All assigned" check returns False if c0==5.
                    # What if c0=4, c1=0, unassigned=1? Handled.
                    
                    # What if c0=3, c1=0, unassigned=2?
                    # If both unassigned are 0 -> c0=5 (Monochromatic, Bad).
                    # If one 0, one 1 -> c0=4, c1=1 (Mixed, Good).
                    # If both 1 -> c0=3, c1=2 (Mixed, Good).
                    # So we MUST avoid the case where both are 0.
                    # This is a binary constraint, not unit. Backtracking handles it.
                    
            return True

        def backtrack(idx):
            """
            Recursive backtracking.
            idx: index of next vertex to assign.
            """
            if idx == n:
                # All assigned. Check if any edge is monochromatic.
                # Actually, propagation should have caught monochromatic edges 
                # when the last vertex of that edge was assigned.
                # But let's be safe.
                for e in edge_vertices:
                    vals = [assignment[v] for v in e]
                    if all(v == 0 for v in vals) or all(v == 1 for v in vals):
                        return False
                return True
            
            # Try 0
            assignment[idx] = 0
            if propagate():
                if backtrack(idx + 1):
                    return True
            assignment[idx] = -1
            
            # Try 1
            assignment[idx] = 1
            if propagate():
                if backtrack(idx + 1):
                    return True
            assignment[idx] = -1
            
            return False

        return backtrack(0)

    def minimize_edges(n, edges):
        """
        Greedy removal of edges while maintaining non-2-colorability.
        """
        edges = list(edges)
        i = 0
        while i < len(edges):
            # Try removing edges[i]
            new_edges = edges[:i] + edges[i+1:]
            # Check if still not 2-colorable
            if not is_2_colorable(n, new_edges):
                # Still not 2-colorable, so we can keep the removal
                edges = new_edges
                # Do not increment i, as the new edge at i might also be removable
            else:
                # Removal makes it 2-colorable, so restore
                i += 1
        return edges

    start_time = time.time()
    best_edges = None
    best_count = float('inf')
    
    # Parameters for random search
    # Known good constructions are around n=10-15, edges~50
    # We will try a range of n and edge counts
    
    random.seed(42) # For reproducibility in search, though we might want different seeds
    
    # Strategy: Try to find a non-2-colorable hypergraph, then minimize.
    # We'll run several iterations.
    
    for iter_num in range(100):
        if time.time() - start_time > 110:
            break
            
        # Choose n
        # Smaller n means fewer total possible edges, harder to be non-2-colorable?
        # Actually, for fixed n, adding edges makes it MORE likely to be non-2-colorable.
        # For n=9, total edges = 126.
        # For n=10, total edges = 252.
        # For n=11, total edges = 462.
        
        # Let's try n=10 first as it's a sweet spot.
        n = random.choice([10, 11, 12])
        
        # Total possible edges
        total_edges = len(list(itertools.combinations(range(n), 5)))
        
        # Start with a random subset
        # If we start too small, it's likely 2-colorable.
        # If we start too big, it's likely non-2-colorable but we have to remove many.
        # Let's start with ~60 edges.
        target_edges = random.randint(50, 80)
        if target_edges > total_edges:
            target_edges = total_edges
            
        all_possible = list(itertools.combinations(range(n), 5))
        current_edges = random.sample(all_possible, target_edges)
        
        # Check if 2-colorable
        if is_2_colorable(n, current_edges):
            # It is 2-colorable. We need to add edges to make it non-2-colorable.
            # Add random edges until non-2-colorable or we run out.
            remaining = [e for e in all_possible if e not in current_edges]
            random.shuffle(remaining)
            
            added = 0
            # Cap additions to avoid infinite loops if it never becomes non-2-colorable (unlikely for high density)
            for e in remaining:
                if time.time() - start_time > 110:
                    break
                current_edges.append(e)
                added += 1
                # Check periodically to save time? Or every time?
                # Every time is safer but slower.
                if not is_2_colorable(n, current_edges):
                    break
                if added > 20: # Heuristic: if we added 20 and still colorable, maybe this n is too small or bad luck?
                    # Actually, if it's still colorable, we should keep adding.
                    # But if we run out of edges, it's still colorable.
                    pass
            
            if is_2_colorable(n, current_edges):
                # Still colorable even with many edges? Try next iteration.
                continue
        
        # Now current_edges is NOT 2-colorable.
        # Minimize it.
        minimized = minimize_edges(n, current_edges)
        
        if len(minimized) < best_count:
            best_count = len(minimized)
            best_edges = minimized
            # Print progress (optional, but good for debugging if allowed)
            # print(f"Iter {iter_num}: n={n}, edges={best_count}")
            
        # Early termination if we hit the theoretical lower bound or known best
        if best_count <= 51:
            # We found a solution as good as the known best. 
            # We could try to go lower, but 51 is the "best known". 
            # The task says "Minimize", so we should keep trying if time permits.
            # But if we hit 40, that would be a breakthrough.
            # Let's keep searching for a bit longer to see if we can go below 51.
            if best_count <= 45:
                break

    if best_edges is None:
        # Fallback: Return a known valid instance if search failed
        # Construct a simple non-2-colorable one? 
        # E.g., all 5-subsets of 9 vertices (126 edges).
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': n, 'edges': [list(e) for e in best_edges]}
