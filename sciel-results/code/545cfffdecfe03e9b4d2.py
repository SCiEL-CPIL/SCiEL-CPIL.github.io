import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Precompute all triples and their bitmasks
    points = range(16)
    all_triples = list(itertools.combinations(points, 3))
    triple_masks = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Map triple index to mask for quick lookup
    # We need to check if two triples share > 1 point.
    # This is equivalent to (mask1 & mask2) having more than 1 bit set.
    # popcount(mask1 & mask2) > 1
    
    def popcount(x):
        return bin(x).count('1')
        
    best_system = []
    best_alpha = 17  # Initialize to impossible high value
    
    # Helper to calculate independence number
    def calc_alpha(system_masks):
        # system_masks is a list of bitmasks for the triples in the system
        # We want the largest k such that there exists a subset of points of size k
        # that does not contain any triple from system_masks.
        
        # Check from k=16 down to 1
        for k in range(16, 0, -1):
            # Iterate through all subsets of size k
            for subset in itertools.combinations(points, k):
                subset_mask = 0
                for p in subset:
                    subset_mask |= (1 << p)
                
                # Check if any triple is contained in this subset
                contains_triple = False
                for t_mask in system_masks:
                    if (t_mask & subset_mask) == t_mask:
                        contains_triple = True
                        break
                if not contains_triple:
                    return k
        return 0

    # Randomized search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Shuffle triples
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        
        current_system_masks = []
        current_triples = []
        
        for idx in indices:
            mask = triple_masks[idx]
            triple = all_triples[idx]
            
            # Check conflict with existing triples
            conflict = False
            for existing_mask in current_system_masks:
                if popcount(mask & existing_mask) > 1:
                    conflict = True
                    break
            
            if not conflict:
                current_system_masks.append(mask)
                current_triples.append(triple)
        
        # Evaluate
        alpha = calc_alpha(current_system_masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(t) for t in current_triples]
            
            # If we hit the theoretical minimum (likely 5 or 6), we can stop early?
            # But maybe 4 is possible? Unlikely for v=16.
            # Let's keep searching to be sure, but if alpha <= 5, it's very good.
            if alpha <= 5:
                # We might find 5 quickly. Let's do a few more to see if 4 is possible (highly unlikely)
                # Or just break if we are confident.
                # For now, keep searching for a bit longer to ensure stability, or break if time is tight.
                pass

    return best_system
