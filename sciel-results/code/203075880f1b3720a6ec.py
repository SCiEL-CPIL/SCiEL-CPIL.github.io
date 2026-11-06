import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin before 150s kill

    # Use n=9, complete 5-uniform hypergraph
    n = 9
    all_edges = list(itertools.combinations(range(n), 5))
    
    # Pre-compute masks for all edges
    edge_masks = []
    for e in all_edges:
        em = 0
        for v in e:
            em |= (1 << v)
        edge_masks.append(em)
        
    # Map mask to edge (assuming unique masks for distinct edges, which is true for sets)
    # Actually, different edges have different masks.
    mask_to_edge = {em: e for e, em in zip(all_edges, edge_masks)}

    def is_not_2colorable_masks(masks):
        """
        Checks if the hypergraph defined by the list of edge masks is NOT 2-colorable.
        Returns True if NOT 2-colorable (UNSAT), False if 2-colorable (SAT).
        Optimized for n=9 (2^8 = 256 colorings to check).
        """
        if time.time() > deadline:
            return True # Conservative
        
        if not masks:
            return False
            
        # Fix vertex 0 to 0. Check all 2^(n-1) = 256 masks for vertices 1..8.
        # The mask passed in `color_mask` corresponds to bits 0..7 for vertices 1..8.
        # Vertex 0 is always 0.
        for color_mask in range(256):
            ok = True
            for em in masks:
                # Full mask for this coloring:
                # Bit 0 is 0.
                # Bits 1..8 are from color_mask bits 0..7.
                # So full_mask = color_mask.
                # Check if edge is monochromatic:
                # (full_mask & em) == 0  => All vertices in edge are 0 (since bit 0 is 0, and any 1s in em must be 0 in full_mask)
                # (full_mask & em) == em => All vertices in edge are 1 (since bit 0 is 0, em must have bit 0 as 0, and all 1s in em must be 1 in full_mask)
                
                # Wait, if em has bit 0 set (vertex 0 in edge), and full_mask bit 0 is 0:
                # (full_mask & em) will have bit 0 as 0.
                # If em has bit 0 set, (full_mask & em) != em (because em has 1 at bit 0, full_mask has 0).
                # So (full_mask & em) == em is False.
                # If em has bit 0 set, (full_mask & em) == 0 is False (unless em is 0, which is impossible).
                # So if vertex 0 is in the edge, it cannot be monochromatic 1 (because v0=0).
                # It can only be monochromatic 0 if all other vertices are 0.
                # Condition for mono 0: (full_mask & em) == 0.
                # This implies all bits in em that are 1 must be 0 in full_mask.
                # Since v0 is 0 in full_mask, if em has v0, it's fine for the 0-check.
                
                # Let's just use the standard check:
                # Edge is mono if all vertices have color 0 OR all vertices have color 1.
                # Color 0: (color_mask & em) == 0 (assuming v0=0, so if v0 in em, it's 0, consistent)
                # Color 1: (color_mask & em) == em AND (em & 1) == 0 (v0 must not be in edge for it to be all 1s, since v0=0)
                
                # Actually, simpler:
                # Let's construct the full mask for the coloring.
                # full_mask = color_mask (bits 0-7 correspond to v1-v8). Bit 8 (v9? No, v0) is 0.
                # Wait, my mapping:
                # color_mask bit i (0-7) corresponds to vertex i+1.
                # Vertex 0 is 0.
                # So full_mask = color_mask.
                
                # Check mono 0: (full_mask & em) == 0
                # Check mono 1: (full_mask & em) == em
                
                # This logic holds if em bit 0 is 0 for mono 1 check?
                # If em bit 0 is 1, (full_mask & em) has bit 0 as 0. em has bit 0 as 1.
                # So (full_mask & em) != em. Correct.
                # If em bit 0 is 0, (full_mask & em) can be equal to em if all 1s in em are 1 in full_mask. Correct.
                
                if (color_mask & em) == 0 or (color_mask & em) == em:
                    ok = False
                    break
            if ok:
                return False # Found a valid coloring
        
        return True # No valid coloring found

    # Greedy Minimization
    best_masks = edge_masks[:]
    best_count = len(best_masks)
    
    # Restart loop
    for r in range(500):
        if time.time() > deadline - 5:
            break
            
        current_masks = best_masks[:]
        random.shuffle(current_masks)
        
        # Greedy removal
        i = 0
        while i < len(current_masks):
            if time.time() > deadline - 2:
                break
            
            # Try removing current_masks[i]
            # Create a new list without this mask
            # Optimization: We can use a list slice
            trial_masks = current_masks[:i] + current_masks[i+1:]
            
            if is_not_2colorable_masks(trial_masks):
                # Removal successful
                current_masks = trial_masks
                # Do not increment i, check the next edge (which is now at index i)
            else:
                # Removal failed, keep the edge
                i += 1
                
        if len(current_masks) < best_count:
            best_count = len(current_masks)
            best_masks = current_masks[:]
            
    # Convert best_masks back to edges
    best_edges = [mask_to_edge[em] for em in best_masks]
    
    return {'n': n, 'edges': [list(e) for e in best_edges]}
