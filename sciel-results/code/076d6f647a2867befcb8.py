import random
import time
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Checks if a 5-uniform hypergraph is 2-colorable.
    Uses brute force for n <= 15, which is fast enough for small m.
    """
    if not edges:
        return True
        
    # Convert edges to bitmasks for faster checking
    # e_mask has bits set for vertices in the edge
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # For n <= 15, 2^n is manageable
    if n <= 15:
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                # Check if edge is monochromatic
                # An edge is monochromatic if all its vertices have the same color in 'mask'
                # We can check: (mask & em) == 0 (all 0) or (mask & em) == em (all 1)
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # Fallback for larger n (unlikely to be needed for minimal search)
        # Use backtracking if n is large, but for this task n will be small
        # We'll just return False to be safe if we can't verify quickly, 
        # but the search will focus on small n.
        return False 

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Initialize with K9^5 (126 edges) as a safe fallback
    n_start = 9
    edges_start = [list(e) for e in combinations(range(n_start), 5)]
    if not is_2_colorable_fast(n_start, edges_start):
        best_edges = edges_start[:]
        best_m = len(edges_start)
        
    # Search for smaller hypergraphs
    # Try n from 9 to 14
    for n in range(9, 15):
        if time.time() > deadline - 5:
            break
            
        # Estimate lower bound for m
        # For 5-uniform, m(5) is around 29. 
        # But for small n, it might be higher.
        # Let's try a range of m values.
        # We can start from a reasonable guess and go up.
        # Or start from high and go down? 
        # Random generation of non-2-colorable is easier with higher density.
        # Let's try to find one with m < best_m.
        
        # We'll try random sampling for various m
        # If n=9, max m is 126.
        # If n=10, max m is 252.
        
        # To find a sparse one, we might need to start dense and prune.
        # Or just random search.
        
        # Let's try to construct by adding edges to a base?
        # Better: Random hypergraph generation.
        
        # We want m < best_m.
        # Let's try m from 30 to best_m - 1
        # But random hypergraphs with low m are likely 2-colorable.
        # So we should start with higher m and try to prune.
        
        # Strategy: 
        # 1. Generate a random dense hypergraph (e.g., 50% of all edges).
        # 2. Check if non-2-colorable.
        # 3. If yes, try to remove edges greedily.
        # 4. If no, try again or increase density.
        
        all_possible = list(combinations(range(n), 5))
        num_edges_total = len(all_possible)
        
        # Try different densities
        for density in [0.8, 0.6, 0.4, 0.2, 0.1]:
            if time.time() > deadline - 5:
                break
            m_target = int(num_edges_total * density)
            if m_target >= best_m:
                continue # We already have a better one
            
            # Try a few random instances
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                random.shuffle(all_possible)
                candidate_edges = [list(e) for e in all_possible[:m_target]]
                
                if not is_2_colorable_fast(n, candidate_edges):
                    # Found a non-2-colorable one. Try to minimize.
                    current_edges = candidate_edges[:]
                    improved = True
                    while improved and time.time() < deadline - 10:
                        improved = False
                        # Try removing edges in random order
                        indices = list(range(len(current_edges)))
                        random.shuffle(indices)
                        for i in indices:
                            if time.time() > deadline - 10:
                                break
                            new_edges = current_edges[:i] + current_edges[i+1:]
                            if not is_2_colorable_fast(n, new_edges):
                                current_edges = new_edges
                                improved = True
                                break
                                
                    if len(current_edges) < best_m:
                        best_m = len(current_edges)
                        best_edges = current_edges[:]
                        best_n = n
                        # If we found a very good one, we can break early?
                        # Maybe continue to see if we can do better.
                        if best_m <= 30: # Close to theoretical lower bound
                             break
            if best_m <= 30:
                break
    if best_edges is None:
        # Fallback
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
