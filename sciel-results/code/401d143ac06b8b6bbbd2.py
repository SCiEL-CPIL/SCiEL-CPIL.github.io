import random
from itertools import combinations
import time

def is_2_colorable(edges, n):
    if not edges:
        return True
        
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Iterate through all 2^n colorings
    for m in range(1 << n):
        valid = True
        for em in edge_masks:
            # Check if edge is monochromatic
            if (m & em) == 0 or (m & em) == em:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    n = 9
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    
    best_edges = all_edges
    best_count = len(all_edges)
    
    start_time = time.time()
    limit = 145.0 # 5s margin
    
    # We can also try to improve the initial set by removing edges in a smart order
    # But random restarts are robust.
    
    for restart in range(1000):
        if time.time() - start_time > limit:
            break
            
        edges = list(all_edges)
        random.shuffle(edges)
        
        kept = set(all_edges)
        
        for e in edges:
            if e not in kept:
                continue
            
            # Check time
            if time.time() - start_time > limit:
                break
                
            kept.discard(e)
            if not is_2_colorable(list(kept), n):
                pass # Keep removed
            else:
                kept.add(e) # Restore
        
        if len(kept) < best_count:
            best_count = len(kept)
            best_edges = list(kept)
            
            # Optional: If we find 51, we can stop or try to find fewer (unlikely)
            if best_count <= 51:
                # We might want to try a bit more to see if we can go lower, 
                # but 51 is the known lower bound for m(5). 
                # Actually, m(5) is the MINIMUM number of edges. 
                # So we cannot go lower than 51.
                # If we find 51, we are optimal.
                # However, we should verify if 51 is indeed the lower bound.
                # The prompt says "best known construction has 51 edges".
                # It doesn't explicitly say 51 is the absolute minimum, 
                # but "m(5)" usually denotes the minimum.
                # "Minimize the number of edges."
                # If I find 51, I should probably stop to save time, 
                # or just let it run to confirm.
                pass

    return {'n': n, 'edges': [list(e) for e in best_edges]}
