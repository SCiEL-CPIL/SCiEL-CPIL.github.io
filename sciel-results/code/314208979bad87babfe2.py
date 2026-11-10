import itertools
import random
import time
import sys

sys.setrecursionlimit(10000)

def _is_non_2_colorable_fast(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no valid coloring exists).
    """
    if not edges:
        return False

    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    full_mask = (1 << n) - 1

    # Precompute vertex degrees for heuristic
    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_degree[idx] += 1
            m ^= lsb

    # Precompute for each edge, the list of vertices (for faster iteration in propagation)
    # Actually, bitmask operations are fast enough.
    
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

                # Check if all 5 assigned
                if assigned_in_edge == mask:
                    # If all color 1 or all color 0, conflict
                    if (mask & ~color1 & full_mask) == 0:  # All color 1
                        return None, None, True
                    if (mask & color1) == 0:  # All color 0
                        return None, None, True
                    continue

                # Check for forcing
                unassigned_in_edge = mask & unassigned
                # If exactly 1 unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    # Are all 4 color 1?
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        # Force to color 0
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
            # Propagation should have caught this, but just in case.
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0:  # All color 1
                    return False
                if (mask & color1) == 0:  # All color 0
                    return False
            return True  # Valid coloring found

        # Pick a vertex to assign: 
        # Heuristic: Find the unassigned vertex that is in the most edges 
        # that have exactly 4 assigned vertices (i.e., it is the last unassigned in those edges).
        # This is a "Most Constrained Variable" variant.
        
        best_v_bit = 0
        best_score = -1
        
        # To speed this up, we could precompute, but n is small.
        # Iterate through unassigned vertices
        m = unassigned
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score = 0
            # Count how many edges this vertex is the "last" unassigned in
            for mask in edge_masks:
                if mask & lsb: # Vertex is in edge
                    un_in_edge = mask & unassigned
                    # If this vertex is the only unassigned in this edge
                    if un_in_edge == lsb:
                        score += 1
            # Tie break with degree
            if score > best_score or (score == best_score and vertex_degree[idx] > vertex_degree[best_v_bit.bit_length() - 1] if best_v_bit else False):
                best_score = score
                best_v_bit = lsb
            m ^= lsb
        
        v_bit = best_v_bit

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

    # Symmetry breaking: Fix vertex 0 to color 0.
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1  # Exclude vertex 0

    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True

    if initial_unassigned == 0:
        for mask in edge_masks:
            if (mask & ~initial_color1 & full_mask) == 0:
                return False
            if (mask & initial_color1) == 0:
                return False
        return False  # Coloring found

    if backtrack(initial_color1, initial_unassigned):
        return False  # Coloring found

    return True  # No coloring found

def _greedy_remove(n, edges, deadline, rng):
    """
    Greedy edge removal with randomized ordering.
    Returns the reduced edge set.
    """
    current_edges = edges[:]
    
    removed_any = True
    while removed_any and (time.time() < deadline):
        removed_any = False
        rng.shuffle(current_edges)

        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                break

            e = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i + 1:]

            if _is_non_2_colorable_fast(n, new_edges):
                current_edges = new_edges
                removed_any = True

    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_edges = None
    best_len = float('inf')
    best_n = 9

    # Try different n values.
    # m(5) is the minimum number of edges in a non-2-colorable 5-uniform hypergraph.
    # It is known that m(5) <= 51.
    # K9 has 126 edges.
    # We expect the minimal non-2-colorable subgraph to have n around 9-12.
    
    for n in range(9, 15):
        if time.time() > deadline - 10: # Reserve 10s for final checks
            break
            
        # Generate all possible edges
        all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        
        # If n is too large, K_n is huge.
        # For n=12, C(12,5) = 792.
        # For n=13, C(13,5) = 1287.
        # For n=14, C(14,5) = 2002.
        
        # We will do multiple restarts for small n, fewer for large n.
        num_restarts = 10 if n <= 10 else 5 if n <= 12 else 2
        
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            rng = random.Random(seed * 100 + n)
            
            reduced = _greedy_remove(n, all_edges, deadline, rng)

            if len(reduced) < best_len:
                best_len = len(reduced)
                best_edges = reduced[:]
                best_n = n

            # If we found a very good solution, we can stop early
            if best_len <= 40:
                break

    # Final verification
    if best_edges is None:
        # Fallback to K9
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    if not _is_non_2_colorable_fast(best_n, best_edges):
        # Fallback to full K9 if something went wrong
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
