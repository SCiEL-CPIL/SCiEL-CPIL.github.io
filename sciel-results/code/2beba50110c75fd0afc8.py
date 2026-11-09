import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Fast 2-colorability checker using backtracking and unit propagation
    def is_2_colorable(n, edges):
        # edges: list of tuples of 5 vertices
        # Precompute edge masks and vertex lists
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # We represent the coloring state as two bitmasks: color0 and color1
        # Unassigned vertices are in neither.
        # We use a recursive function.
        
        # To speed up, we can precompute for each vertex, the edges it belongs to.
        # But with small n, simple backtracking is okay if we prune well.
        
        # Better: Use a constraint satisfaction approach.
        # Variables: vertices 0..n-1.
        # Domain: {0, 1}.
        
        # We will use a simple DFS with unit propagation.
        # State: colors array of size n, -1 for unassigned.
        # To make it fast, we use bitmasks for assigned 0s and 1s.
        
        # Let's use a list of edges as tuples for fast access.
        edges_list = [tuple(e) for e in edges]
        
        # Heuristic: Order edges by "tightness" is hard dynamically.
        # Just iterate.
        
        # Optimization: If n is small, we can even brute force if edges are few? No, 2^n is too big for n=15 (32768 is fine actually).
        # 2^15 = 32768. For each coloring, check all edges.
        # If we have 50 edges, 50 * 32768 = 1.6M ops. This is very fast in Python? Maybe 1-2 seconds.
        # So for n <= 15, we can just brute force all 2^n colorings!
        
        if n <= 15:
            # Brute force
            # Map vertices to bits in an integer coloring mask
            # color 0: bit is 0, color 1: bit is 1?
            # Let's say bit v is 1 if color is 1, 0 if color is 0.
            # For an edge (v1..v5), it is monochromatic if all bits are same.
            # i.e., (mask & edge_mask) == 0 or (mask & edge_mask) == edge_mask
            
            masks = [0] * len(edges_list)
            for i, e in enumerate(edges_list):
                m = 0
                for v in e:
                    m |= (1 << v)
                masks[i] = m
            
            for color_mask in range(1 << n):
                valid = True
                for m in masks:
                    # Check if m is subset of color_mask (all 1s)
                    if (color_mask & m) == m:
                        valid = False
                        break
                    # Check if m is disjoint from color_mask (all 0s)
                    if (color_mask & m) == 0:
                        valid = False
                        break
                if valid:
                    return True # 2-colorable
            return False # Not 2-colorable
        else:
            # Fallback to backtracking for larger n (though we likely won't go > 15)
            # ... implement backtracking ...
            # For now, assume n <= 15
            pass

    # Search for minimal non-2-colorable hypergraphs
    for n in range(9, 16):
        if time.time() > deadline:
            break
            
        # Start with complete hypergraph
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Try multiple random orderings of edge removal
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            current_edges = list(all_edges)
            random.shuffle(current_edges)
            
            # Minimization: try to remove each edge
            # We iterate through the list. If we can remove edge i, we remove it.
            # This is a greedy minimalization.
            
            # To be more effective, we can do multiple passes or random restarts of the removal order.
            # The order matters.
            
            # Let's just try to remove edges in the shuffled order.
            new_edges = []
            for e in current_edges:
                # Try removing e
                # Create a list without e? That's O(E^2).
                # Better: keep a list, and mark removed.
                pass
            
            # Efficient minimization:
            # Start with full set.
            # For each edge in random order:
            #   Temporarily remove.
            #   Check 2-colorability.
            #   If 2-colorable, restore.
            #   If not, keep removed.
            
            edges_to_check = list(all_edges)
            random.shuffle(edges_to_check)
            
            kept_edges = []
            # We maintain the current set of kept edges.
            # Initially empty? No, we start with all, and remove.
            # Or start with empty and add?
            # "Minimizing" usually means starting dense and removing.
            
            # Let's start with all edges, and try to remove.
            current_set = set(all_edges)
            
            # This is O(E * Cost_Check).
            # E can be up to 3003 (for n=15).
            # Cost_Check is ~ 1ms (brute force 2^15 * 50 edges).
            # 3000 * 1ms = 3 seconds per attempt.
            # 50 attempts * 7 values of n = 350 seconds. Too slow.
            
            # We need to be smarter.
            # 1. Reduce attempts.
            # 2. Use a faster checker?
            #    Brute force 2^15 is 32768 iterations.
            #    In Python, a loop of 32768 with some bit ops is ~ 5-10 ms.
            #    So 3000 * 10ms = 30 seconds per full minimization.
            #    This is too slow for 50 attempts.
            
            # Strategy:
            # Only do full minimization for promising n.
            # Or, use a heuristic to find a small set first, then minimize that.
            
            # Alternative: Random Subgraph Search
            # Pick a random subset of edges of size K.
            # Check if non-2-colorable.
            # If yes, minimize it.
            # If no, increase K or pick different subset.
            
            # What K to start with?
            # For n=15, m(5) <= 51.
            # So K=51 is a good target.
            # Let's try to find a 51-edge non-2-colorable hypergraph.
            
            # Randomly sample 51 edges from K_15^(5).
            # Check if non-2-colorable.
            # Probability is low?
            # Maybe use a biased sampling?
            
            # Actually, let's stick to the minimization of the complete graph but optimize the checker.
            # If we use a C-like speed? No, Python.
            
            # Let's reduce the number of attempts and n values.
            # Focus on n=15, as it's the known sweet spot.
            # And maybe n=14, 13.
            
            # Let's try a different approach:
            # Start with a known small non-2-colorable hypergraph if possible?
            # I don't have one hardcoded.
            
            # Let's try to minimize the complete graph on n=9 first.
            # n=9, E=126.
            # 126 checks * 1ms = 0.126s.
            # 100 attempts = 12s.
            # This is feasible.
            
            # Then n=10, E=252.
            # 252 * 2ms (2^10 is small, so checker is faster? No, brute force 2^10=1024 is very fast. <1ms)
            # 252 * 0.5ms = 0.12s.
            # 100 attempts = 12s.
            
            # n=11, E=462.
            # 2^11 = 2048. Fast.
            # 462 * 0.5ms = 0.2s.
            # 50 attempts = 10s.
            
            # n=12, E=792.
            # 2^12 = 4096.
            # 792 * 1ms = 0.8s.
            # 20 attempts = 16s.
            
            # n=13, E=1287.
            # 2^13 = 8192.
            # 1287 * 2ms = 2.5s.
            # 5 attempts = 12.5s.
            
            # n=14, E=2002.
            # 2^14 = 16384.
            # 2002 * 4ms = 8s.
            # 2 attempts = 16s.
            
            # n=15, E=3003.
            # 2^15 = 32768.
            # 3003 * 8ms = 24s.
            # 1 attempt = 24s.
            
            # Total ~ 12+12+10+16+12+16+24 = 102s.
            # This fits in 110s!
            
            # Let's implement this distribution of attempts.
            
            pass # Placeholder for logic below

    # Re-implementing the search loop with specific budget allocation
    budgets = {
        9: 100,
        10: 100,
        11: 50,
        12: 20,
        13: 5,
        14: 2,
        15: 1
    }
    
    for n in sorted(budgets.keys()):
        attempts = budgets[n]
        all_edges = list(combinations(range(n), 5))
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Shuffle edges to randomize removal order
            shuffled_edges = list(all_edges)
            random.shuffle(shuffled_edges)
            
            # Start with all edges
            current_edges = set(shuffled_edges)
            
            # Try to remove each edge
            # We iterate through the shuffled list.
            # If we remove an edge, we don't add it back.
            # We check if the *current* set minus the edge is 2-colorable.
            
            # Optimization: If the current set is already 2-colorable, we stop (shouldn't happen if we start dense and only remove if still non-2-colorable).
            
            for e in shuffled_edges:
                if e not in current_edges:
                    continue
                
                # Try removing e
                temp_edges = current_edges - {e}
                
                # Check 2-colorability
                # Convert to list for checker
                edge_list = list(temp_edges)
                
                if is_2_colorable(n, edge_list):
                    # It became 2-colorable, so e is necessary
                    # Keep e (do nothing, it's already in current_edges)
                    pass
                else:
                    # It is still non-2-colorable, so e is redundant
                    current_edges.discard(e)
            
            # End of minimization
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                best_n = n
                # Print progress? No, just update.
                # If we find 51, we can stop early?
                # The task says "Minimize". We should try to beat 51 if possible.
                # But 51 is the best known.
                # If we find 51, it's a huge improvement over 126.
                
                if best_count <= 51:
                    # We hit the known lower bound (or close). 
                    # We can break out of all loops to save time?
                    # Or continue to see if we can find < 51?
                    # "best known construction has 51 edges".
                    # It is unlikely we find < 51.
                    # Let's break to return safely.
                    break
        
        if best_count <= 51:
            break
    
    if best_edges is None:
        # Fallback to seed
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}

# Helper for the checker inside construct
# (Defined inside construct to avoid global state issues, though global is fine too)
# The is_2_colorable function defined above uses brute force for n<=15.
# Let's refine the brute force to be as fast as possible.

def _fast_is_2_colorable(n, edges):
    if n <= 15:
        # Precompute masks
        masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            masks.append(m)
        
        # Iterate all colorings
        # Use a trick: if n is small, 1<<n is small.
        for color_mask in range(1 << n):
            # Check each edge
            # An edge is monochromatic if (color_mask & m) == m or (color_mask & m) == 0
            ok = True
            for m in masks:
                if (color_mask & m) == m or (color_mask & m) == 0:
                    ok = False
                    break
            if ok:
                return True
        return False
    else:
        # Fallback: simple backtracking (not implemented fully, but n will be <= 15)
        return True # Assume 2-colorable if we can't check? No, this is dangerous.
        # But we only call this for n <= 15.

# Replace the inner is_2_colorable with _fast_is_2_colorable
# Let's rewrite construct to use _fast_is_2_colorable

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    budgets = {
        9: 100,
        10: 100,
        11: 50,
        12: 20,
        13: 5,
        14: 2,
        15: 1
    }
    
    for n in sorted(budgets.keys()):
        attempts = budgets[n]
        all_edges = list(combinations(range(n), 5))
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            shuffled_edges = list(all_edges)
            random.shuffle(shuffled_edges)
            
            current_edges = set(shuffled_edges)
            
            for e in shuffled_edges:
                if e not in current_edges:
                    continue
                
                temp_edges = current_edges - {e}
                edge_list = list(temp_edges)
                
                if _fast_is_2_colorable(n, edge_list):
                    pass
                else:
                    current_edges.discard(e)
            
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                best_n = n
                
                if best_count <= 51:
                    break
        
        if best_count <= 51:
            break
    
    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
