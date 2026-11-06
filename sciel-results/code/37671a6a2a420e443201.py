import time
import itertools

def construct():
    start = time.time()
    deadline = start + 125  # Safety margin before 150s kill

    def is_not_2colorable_fast(n, edge_masks):
        """
        Check if hypergraph defined by n vertices and edge_masks is NOT 2-colorable.
        Uses bitmask brute-force for small n.
        Returns True if NOT 2-colorable (UNSAT), False if 2-colorable (SAT).
        """
        if n > 20:
            # Fallback to slower method or assume SAT for safety in search?
            # For now, we only use this for n <= 12 or so.
            pass
            
        m = len(edge_masks)
        if m == 0:
            return False

        # Brute force all 2^n colorings
        # Symmetry: Fix vertex 0 to 0. So we only check 2^(n-1) colorings.
        # Actually, just check all 2^n, it's fast enough for n=9,10.
        for mask in range(1 << n):
            is_valid_coloring = True
            for em in edge_masks:
                # Edge is monochromatic if (mask & em) == 0 or (mask & em) == em
                if (mask & em) == 0 or (mask & em) == em:
                    is_valid_coloring = False
                    break
            if is_valid_coloring:
                return False # Found a valid coloring, so it IS 2-colorable
        
        return True # No valid coloring found, so NOT 2-colorable

    def edges_to_masks(edges):
        masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            masks.append(m)
        return masks

    # Strategy: Start with complete 5-uniform hypergraph on n vertices, greedily remove edges.
    best_n = 9
    best_edges = []
    best_m = float('inf')

    # Try n=9 first, as it has the smallest complete hypergraph
    for n_try in [9, 10, 11]:
        if time.time() > deadline - 20:
            break

        # Generate all 5-subsets
        all_edges = list(itertools.combinations(range(n_try), 5))
        current_edges = all_edges[:]
        
        # Convert to masks for fast checking
        # We need to update masks as we remove edges.
        # It's easier to keep list of edges and convert to masks each time?
        # No, converting 126 edges is fast.
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 10:
                break
                
            # Try to remove each edge
            for i in range(len(current_edges) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                    
                # Candidate hypergraph without edge i
                candidate_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if candidate is non-2-colorable
                masks = edges_to_masks(candidate_edges)
                if is_not_2colorable_fast(n_try, masks):
                    # It is still non-2-colorable, so we can remove this edge
                    current_edges = candidate_edges
                    improved = True
                    # Restart the loop to check from the end again?
                    # Or continue? Restarting is safer to ensure minimality.
                    # But it's slower.
                    # Since we removed one, we can just continue to next i (which is now i-1 effectively)
                    # Actually, just break and restart the while loop.
                    break

        if len(current_edges) < best_m:
            best_m = len(current_edges)
            best_edges = current_edges[:]
            best_n = n_try

    # Final check and formatting
    if best_edges:
        # Determine n from the vertices used (should be best_n)
        max_v = max(max(e) for e in best_edges) + 1
        n_final = max(5, max_v)
        edges_final = [list(e) for e in best_edges]
        return {'n': n_final, 'edges': edges_final}

    # Fallback: Complete hypergraph on 9 vertices
    n = 9
    edges = [list(e) for e in itertools.combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
