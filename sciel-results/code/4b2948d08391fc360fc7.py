import itertools
import random
import time

def _is_non_2_colorable_fast(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no valid coloring exists).
    """
    if not edges:
        return False

    # Map vertex to index 0..n-1 (already 0-indexed)
    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    # We search for a coloring that avoids monochromatic edges.
    # A coloring is a pair of sets (S, T) where S is color 0, T is color 1, S union T = V, S intersect T = empty.
    # Equivalently, assign each vertex 0 or 1.
    # Let's use a recursive search on vertices.
    # To speed up, we use bitmasks for assigned vertices and colors.
    # assigned_mask: bits for vertices that are assigned.
    # color_mask: bits for vertices assigned color 1. (Vertices in assigned_mask but not in color_mask are color 0).
    
    # Unit Propagation Logic:
    # If an edge has 4 vertices assigned color 0, the 5th MUST be color 1.
    # If an edge has 4 vertices assigned color 1, the 5th MUST be color 0.
    # If an edge has all 5 assigned same color -> Conflict.

    # Precompute for each edge, the list of vertices? Or just use masks.
    # Let's define a helper to check if an edge is "satisfied" or "conflict" or "forcing".
    
    # We will iterate vertices in a fixed order, but use propagation to skip.
    # Actually, a simpler robust approach for n<=26 and small edge count:
    # Recursive search on the smallest unassigned vertex.
    
    # State:
    # color0: bitmask of vertices colored 0
    # color1: bitmask of vertices colored 1
    # unassigned: bitmask of vertices not yet colored
    
    # Optimization: We can represent the state as (color1, unassigned). color0 is derived?
    # No, color0 = ~color1 & ~unassigned & (1<<n)-1.
    
    # Let's use a simple recursive function.
    
    full_mask = (1 << n) - 1
    
    def propagate(color1, unassigned):
        """
        Apply unit propagation. Returns (new_color1, new_unassigned, conflict_flag).
        If conflict, returns (None, None, True).
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                # Intersect with assigned vertices
                assigned_in_edge = mask & ~unassigned
                # Count bits in assigned_in_edge & color1 and assigned_in_edge & ~color1
                c1_in_edge = assigned_in_edge & color1
                c0_in_edge = assigned_in_edge & ~color1 & full_mask # Note: ~color1 needs masking
                
                # Actually, let's just compute:
                # v1 = number of vertices in edge with color 1
                # v0 = number of vertices in edge with color 0
                # vu = number of vertices in edge unassigned
                
                # Efficient bit counting:
                # bin(x).count('1') is slow in tight loops? Maybe.
                # But n is small.
                
                # Let's check specific cases:
                # Case 1: All 5 assigned and all color 1 -> Conflict
                if assigned_in_edge == mask:
                    if (mask & ~color1 & full_mask) == 0: # All are color 1
                        return None, None, True
                    if (mask & color1) == 0: # All are color 0
                        return None, None, True
                    
                # Case 2: 4 assigned color 1, 1 unassigned -> Force unassigned to color 0
                # Case 3: 4 assigned color 0, 1 unassigned -> Force unassigned to color 1
                
                # Let's identify the unassigned vertices in the edge
                unassigned_in_edge = mask & unassigned
                # If there is exactly 1 unassigned vertex
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    # It's a single bit
                    # Check the other 4
                    other_4 = mask ^ unassigned_in_edge
                    # Are all 4 color 1?
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        # Force to color 0
                        # Just don't set it in color1. It's already not in color1.
                        # Remove from unassigned
                        unassigned ^= unassigned_in_edge
                        changed = True
                    # Are all 4 color 0?
                    elif (other_4 & color1) == 0:
                        # Force to color 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True
                        
        return color1, unassigned, False

    def backtrack(color1, unassigned):
        # Check for immediate conflict via propagation
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        
        if unassigned == 0:
            # All assigned. Check if any edge is monochromatic.
            # Propagation should have caught monochromatic edges if they were fully assigned.
            # But let's double check just in case propagation logic has gaps.
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0: # All color 1
                    return False
                if (mask & color1) == 0: # All color 0
                    return False
            return True # Valid coloring found
            
        # Pick a vertex to assign.
        # Heuristic: Pick vertex with highest degree in remaining unassigned graph?
        # Or just the lowest bit.
        # Let's pick the lowest bit for determinism, or maybe one involved in most edges.
        # For simplicity, lowest bit.
        v_bit = unassigned & -unassigned
        v_idx = v_bit.bit_length() - 1
        
        # Try color 1
        new_color1 = color1 | v_bit
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned):
            return True
            
        # Try color 0
        new_color1 = color1
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned):
            return True
            
        return False

    # Start search
    # Symmetry breaking: Fix vertex 0 to color 0.
    # If a coloring exists, one with v0=0 exists (by swapping colors if needed).
    # So we only need to search v0=0.
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1 # Exclude vertex 0
    
    # Propagate initial state
    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True # No coloring with v0=0, and by symmetry no coloring at all?
        # Wait. If v0=0 leads to conflict, does it mean no coloring?
        # If there is a valid coloring, either v0=0 or v0=1.
        # If v0=1, swap colors -> v0=0.
        # So if v0=0 branch fails, no coloring exists.
        return True
        
    if initial_unassigned == 0:
        # Check validity
        for mask in edge_masks:
            if (mask & ~initial_color1 & full_mask) == 0:
                return False
            if (mask & initial_color1) == 0:
                return False
        return False # Coloring found
        
    if backtrack(initial_color1, initial_unassigned):
        return False # Coloring found
        
    return True # No coloring found

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin

    # Start with K9^(5)
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Optimize: We want to remove edges.
    # We use a greedy approach: try to remove each edge.
    # To make it faster, we can order edges by some heuristic?
    # Or just random shuffle.
    
    # We will loop until no more edges can be removed.
    # Since the checker is fast, we can do many iterations.
    
    # To avoid getting stuck in local minima, we can restart with different shuffles
    # if we have time. But greedy removal is usually good for this size.
    
    removed_any = True
    while removed_any and (time.time() - start_time) < deadline:
        removed_any = False
        # Shuffle edges to try removing in random order
        random.shuffle(current_edges)
        
        # Iterate backwards to allow removal
        for i in range(len(current_edges) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
                
            e = current_edges[i]
            # Try removing e
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if new_edges is non-2-colorable
            if _is_non_2_colorable_fast(n, new_edges):
                # It is still non-2-colorable, so we can remove e
                current_edges = new_edges
                removed_any = True
            # Else, keep e
            
        # If we removed any, loop again to see if more can be removed
        # (Removing one edge might make another redundant)
        
    # Final verification
    # Ensure the result is valid
    if not _is_non_2_colorable_fast(n, current_edges):
        # Fallback to full K9 if something went wrong (shouldn't happen)
        return {'n': n, 'edges': all_edges}
        
    return {'n': n, 'edges': current_edges}
