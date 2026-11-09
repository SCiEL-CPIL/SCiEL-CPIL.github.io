import itertools
import time

def construct(v, k, t):
    # Precompute bitmask for any tuple of points
    def tuple_to_mask(pts):
        mask = 0
        for p in pts:
            mask |= (1 << p)
        return mask

    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    all_t_masks = [tuple_to_mask(sub) for sub in all_t_subs]
    
    # Precompute for each k-subset, the list of t-subset masks it contains
    # This allows fast gain calculation
    block_to_t_masks = {}
    for block in itertools.combinations(range(v), k):
        b_mask = tuple_to_mask(block)
        t_masks_in_block = []
        for t_sub in itertools.combinations(block, t):
            t_masks_in_block.append(tuple_to_mask(t_sub))
        block_to_t_masks[b_mask] = t_masks_in_block
        
    # Initialize uncovered set
    uncovered = set(all_t_masks)
    blocks = []
    
    start_time = time.time()
    
    while uncovered:
        # Deadline guard
        if time.time() - start_time > 115:
            break
            
        # Pick an arbitrary uncovered t-subset
        s_mask = next(iter(uncovered))
        
        # Decode s_mask to points to generate candidates
        s_points = [i for i in range(v) if (s_mask >> i) & 1]
        remaining_points = [p for p in range(v) if p not in s_points]
        
        best_block_mask = None
        best_gain = -1
        
        # Generate all k-subsets containing s
        # We need to choose (k - t) more points from remaining_points
        for extra in itertools.combinations(remaining_points, k - t):
            b_points = tuple(sorted(s_points + list(extra)))
            b_mask = tuple_to_mask(b_points)
            
            # Calculate gain: how many t-subsets of this block are in uncovered
            # block_to_t_masks[b_mask] is a list of masks
            gain = 0
            for m in block_to_t_masks[b_mask]:
                if m in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_mask = b_mask
                
        if best_block_mask is not None:
            blocks.append(tuple_to_points(best_block_mask)) # Return as list of ints
            # Remove covered t-subsets
            for m in block_to_t_masks[best_block_mask]:
                uncovered.discard(m)
                
    return [list(b) for b in blocks]

# Helper to convert mask back to list of points
def tuple_to_points(mask):
    return [i for i in range(19) if (mask >> i) & 1]
