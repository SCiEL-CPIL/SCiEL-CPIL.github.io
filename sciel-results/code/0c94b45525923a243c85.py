import random
import time
from itertools import combinations

def is_2_colorable_fast(n, edge_masks):
    for mask in range(1 << n):
        is_good = True
        for em in edge_masks:
            if (mask & em) == 0 or (mask & em) == em:
                is_good = False
                break
        if is_good:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = None
    best_len = float('inf')
    best_n = 9
    
    for n in [8, 9, 10]:
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        edge_masks = [sum(1 << v for v in e) for e in all_edges]
        
        # Try random removal
        # Number of iterations depends on n
        max_iters = 500 if n == 8 else 200 if n == 9 else 50
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            random.shuffle(all_edges)
            random.shuffle(edge_masks)
            
            # Use a set of indices for current edges
            current_indices = set(range(len(all_edges)))
            
            # We need to map shuffled edges to original masks
            # Actually, just shuffle the list of masks and indices together?
            # Or just shuffle the list of edges, and recompute masks?
            # Let's shuffle the list of (edge, mask)
            pairs = list(zip(all_edges, edge_masks))
            random.shuffle(pairs)
            
            current_pairs = pairs[:]
            # We want to remove edges.
            # Start with all.
            active_masks = [p[1] for p in current_pairs]
            
            for idx in range(len(current_pairs)):
                if time.time() > deadline:
                    break
                # Try removing edge at idx
                # We are iterating through the shuffled list
                # We remove it from active set
                # But we need to check if the REMAINING set is non-2-colorable
                # This is O(E * 2^n)
                
                # Optimization: Only check if removal is "safe"?
                # No, we must check.
                
                # To avoid O(E) copy, we can just skip the edge in the check?
                # Or maintain a list of active masks.
                
                # Let's maintain a list of active masks.
                # Initially all masks.
                # We iterate through the shuffled order.
                # If we remove it, we check the rest.
                
                pass # Logic below
                
            # Better logic:
            active_masks = [p[1] for p in pairs]
            # We will iterate through the shuffled order of indices
            # and try to remove them.
            # But we need to know which ones are still active.
            
            # Let's use a boolean array for active status
            active = [True] * len(pairs)
            
            for idx in range(len(pairs)):
                if time.time() > deadline:
                    break
                if not active[idx]:
                    continue
                    
                # Try removing idx
                active[idx] = False
                
                # Build list of active masks
                # This is O(E)
                current_masks = [pairs[i][1] for i in range(len(pairs)) if active[i]]
                
                if not is_2_colorable_fast(n, current_masks):
                    # Still non-2-colorable, keep removed
                    continue
                else:
                    # Becomes 2-colorable, put back
                    active[idx] = True
            
            # End of pass
            final_masks = [pairs[i][1] for i in range(len(pairs)) if active[i]]
            if not is_2_colorable_fast(n, final_masks):
                if len(final_masks) < best_len:
                    best_len = len(final_masks)
                    best_edges = [pairs[i][0] for i in range(len(pairs)) if active[i]]
                    best_n = n
                    print(f"Found non-2-colorable with {best_len} edges on n={best_n} at iter {i}")
                    if best_len <= 51:
                        # We can break early for this n
                        break
    
    if best_edges is None:
        # Fallback
        n = 9
        all_edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': all_edges}
        
    return {'n': best_n, 'edges': best_edges}
