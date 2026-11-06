import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def is_not_2_colorable(edge_masks, n):
        """
        Returns True if the hypergraph is NOT 2-colorable.
        edge_masks: list of integers representing edges as bitmasks.
        n: number of vertices.
        """
        # Brute force check all 2^n colorings
        # Optimization: If n is small, this is fast.
        # We can skip C=0 and C=(1<<n)-1 as they are always mono for any edge.
        for c in range(1, (1 << n) - 1):
            is_valid = True
            for em in edge_masks:
                # Check if edge is monochromatic
                # Mono if (c & em) == 0 or (c & em) == em
                if (c & em) == 0 or (c & em) == em:
                    is_valid = False
                    break
            if is_valid:
                return False # Found a valid 2-coloring
        return True # No valid 2-coloring found

    best_m = 200  # Max allowed is 200, start high
    best_result = None

    # Search for m from 29 up to 51 (known best)
    # If we find one < 51, that's a breakthrough.
    # If we find one == 51, that's the known best.
    
    # We'll iterate m from 29 to 51
    for m in range(29, 52):
        if time.time() > deadline - 10:
            break
            
        # Try different n values
        # n must be at least 5.
        # For small m, n is small.
        # Let's try n from 10 to 15 first, as 51 edges on 10-15 vertices is plausible.
        # Actually, for m=51, n=10 has 252 edges total. 51 is ~20%.
        # n=15 has 3003 edges. 51 is ~1.7%.
        # Non-2-colorability is a "global" property.
        
        found_for_m = False
        for n in range(10, 16):
            if time.time() > deadline - 5:
                break
                
            all_edges = list(itertools.combinations(range(n), 5))
            if len(all_edges) < m:
                continue
                
            # Convert all edges to masks for faster checking
            all_masks = [0] * len(all_edges)
            for i, e in enumerate(all_edges):
                mask = 0
                for v in e:
                    mask |= (1 << v)
                all_masks[i] = mask
            
            # Try random samples
            # Number of trials depends on n. Smaller n -> faster check -> more trials.
            max_trials = 1000 if n <= 12 else 100
            if n == 15: max_trials = 50
            
            for trial in range(max_trials):
                if time.time() > deadline - 2:
                    break
                    
                # Pick m random edges
                # Using random.sample on indices is efficient
                indices = random.sample(range(len(all_edges)), m)
                edge_masks = [all_masks[i] for i in indices]
                
                if is_not_2_colorable(edge_masks, n):
                    # Found a non-2-colorable hypergraph with m edges
                    # Convert back to vertex lists
                    edges = [list(all_edges[i]) for i in indices]
                    best_m = m
                    best_result = {'n': n, 'edges': edges}
                    found_for_m = True
                    break
            
            if found_for_m:
                break
        
        if found_for_m:
            # If we found one for this m, we can stop if m is small enough
            # Or continue to see if we can find smaller? No, we are iterating m upwards.
            # So the first m we find is the smallest.
            break

    if best_result is None:
        # Fallback: Return a known 51-edge construction if we didn't find one via search
        # Or just return a valid one with more edges to be safe?
        # The prompt says "best known construction has 51 edges".
        # If we didn't find one, we might fail to score well.
        # Let's try a greedy construction or just return a random dense one?
        # No, we must minimize.
        # If search failed, it's likely because 51 is hard to find randomly or my search space was wrong.
        # Let's try one more specific strategy: 
        # Use the Fano plane structure? No, k=3.
        
        # If we really find nothing, we return a hypergraph with 51 edges if we can construct one?
        # Actually, if the search didn't find one, it's possible I missed it.
        # Let's expand the search range for m=51 if we didn't find anything smaller.
        if best_m == 200:
             # Retry m=51 with more trials and different n
             for n in range(10, 21):
                 if time.time() > deadline - 5:
                     break
                 all_edges = list(itertools.combinations(range(n), 5))
                 if len(all_edges) < 51:
                     continue
                 all_masks = [0] * len(all_edges)
                 for i, e in enumerate(all_edges):
                     mask = 0
                     for v in e:
                         mask |= (1 << v)
                     all_masks[i] = mask
                 
                 max_trials = 200 if n <= 15 else 50
                 for trial in range(max_trials):
                     if time.time() > deadline - 2:
                         break
                     indices = random.sample(range(len(all_edges)), 51)
                     edge_masks = [all_masks[i] for i in indices]
                     if is_not_2_colorable(edge_masks, n):
                         edges = [list(all_edges[i]) for i in indices]
                         best_result = {'n': n, 'edges': edges}
                         best_m = 51
                         break
                 if best_result:
                     break

    if best_result is None:
        # Absolute fallback: Return a valid hypergraph, even if not minimal.
        # A complete hypergraph on 9 vertices is non-2-colorable.
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_result = {'n': n, 'edges': edges}

    return best_result
