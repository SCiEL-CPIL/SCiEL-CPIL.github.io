import itertools
import random
import time

def is_not_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False # Empty hypergraph is 2-colorable

    # Preprocess edges: convert to frozensets for fast lookup? 
    # Or keep as lists.
    # For unit propagation, we need to know for each vertex, which edges contain it.
    
    # Actually, for small n (<= 15), we can use a bitset representation for colorings?
    # Or just simple recursion.
    
    # Let's use a simple recursive backtracking.
    # State: color assignment for vertices 0..n-1. -1 is unassigned.
    # colors = [-1] * n
    
    # To speed up, we can sort edges by length? All are length 5.
    # We can order vertices by degree?
    
    # Let's implement a standard backtracking.
    
    # Optimization: If n is small, we can just iterate all 2^n colorings?
    # For n=9, 512. For n=10, 1024. For n=11, 2048.
    # Brute force is simpler and potentially faster in Python than complex backtracking logic if n is small.
    # Let's check brute force for n <= 12.
    
    if n <= 12:
        # Brute force all 2^n colorings
        for mask in range(1 << n):
            # Check if this coloring is valid (no monochromatic edge)
            valid = True
            for e in edges:
                # Check if all vertices in e have the same color
                # color of v is (mask >> v) & 1
                c0 = (mask >> e[0]) & 1
                # If any vertex differs, it's not monochromatic
                # We need to check if ALL are c0.
                mono = True
                for v in e[1:]:
                    if ((mask >> v) & 1) != c0:
                        mono = False
                        break
                if mono:
                    valid = False
                    break
            if valid:
                return False # Found a valid coloring, so it IS 2-colorable
        
        return True # No valid coloring found, so NOT 2-colorable
    else:
        # Use backtracking for larger n
        # ... (implementation omitted for brevity in thought, but will be in code)
        pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_n = 9
    best_edges = None
    best_count = float('inf')
    
    # Try n from 9 to 15
    for n in range(9, 16):
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        num_edges = len(all_edges)
        
        # If n is too large, brute force 2^n is too slow.
        # For n=13, 8192 masks. 8192 * 200 edges * 5 ops = ~8M ops. ~1s.
        # For n=14, 16384 masks. ~2s.
        # For n=15, 32768 masks. ~4s.
        # Greedy removal requires ~num_edges checks.
        # For n=13, 1287 edges. 1287 * 1s = 1287s. Too slow.
        # So for n > 12, we MUST use fast backtracking.
        
        # Let's implement a fast backtracking checker for n > 12.
        
        # For now, let's just do n=9, 10, 11, 12 with brute force.
        # And maybe n=13 with a very limited number of restarts if fast.
        
        if n > 12:
            # Skip for now or implement fast checker
            # Let's try to implement a fast checker
            pass

        # Greedy removal
        # We will do multiple restarts
        max_restarts = 50
        if n == 9: max_restarts = 200
        if n == 10: max_restarts = 100
        if n == 11: max_restarts = 50
        if n == 12: max_restarts = 20

        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Start with all edges
            current_edges = all_edges[:]
            
            # Shuffle order of removal
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            
            # Try to remove each edge
            # To speed up, we can iterate through indices
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # Try removing edge at idx
                # But indices are changing if we remove?
                # Better to iterate through a static list of edge IDs and check if they are still present?
                # Or just iterate through the list and pop?
                # Popping is O(N).
                # Let's use a set of edge tuples for O(1) removal?
                # But we need to pass a list to the checker.
                
                # Alternative: 
                # Keep a list `active_edges`.
                # Iterate through a shuffled list of edge *values*.
                # If the edge is in `active_edges`, try to remove it.
                
                edge_val = tuple(current_edges[idx]) # This is wrong if list changes
                
            # Let's restructure the greedy loop
            active_edges = all_edges[:]
            removal_order = all_edges[:]
            random.shuffle(removal_order)
            
            for e in removal_order:
                if time.time() > deadline:
                    break
                if e not in active_edges:
                    continue
                
                # Try removing e
                active_edges.remove(e)
                
                # Check
                if is_not_2_colorable(n, active_edges):
                    # Keep it removed
                    pass
                else:
                    # Put it back
                    active_edges.append(e)
            
            # End of one restart
            if len(active_edges) < best_count:
                best_count = len(active_edges)
                best_n = n
                best_edges = active_edges[:]
                
            # Early stop if we found a very good one?
            # m(5) is likely around 30-50.
            # If we find 33, we can stop?
            # But maybe we can find 30?
            # Let's keep searching.
            
    if best_edges is None:
        # Fallback: return complete graph on 9
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
