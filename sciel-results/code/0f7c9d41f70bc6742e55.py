"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Best-Fit Decreasing with empty-space management.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin before 120s kill
    
    W, H, D = container
    container_volume = W * H * D
    
    # Sort boxes by volume descending (largest first)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # We'll track bins, each with a list of empty spaces
    # Each empty space is (x, y, z, w, h, d)
    # Start with one bin containing the full container as an empty space
    bins = []
    bin_empty_spaces = [[[0, 0, 0, W, H, D]]]
    bin_volumes = [0]
    
    # Result will be a list of (bin_id, x, y, z, w, h, d)
    # We need to map back to original box order
    placements = [None] * len(boxes)
    
    def split_space(space, box):
        """
        Split an empty space into up to 3 sub-spaces after placing a box.
        space: (x, y, z, w, h, d)
        box: (w, h, d) placed at (x, y, z)
        Returns list of new sub-spaces.
        """
        sx, sy, sz, sw, sh, sd = space
        bw, bh, bd = box
        # Box is placed at (sx, sy, sz) within this space
        # The remaining space is split into up to 3 parts:
        # 1. Above the box: (sx, sy+bh, sz, sw, sh-bh, sd) if sh > bh
        # 2. To the right of the box: (sx+bw, sy, sz, sw-bw, bh, sd) if sw > bw
        # 3. Behind the box: (sx, sy, sz+bd, bw, bh, sd-bd) if sd > bd
        # But we need to be careful: the standard approach is to split along one axis at a time.
        # A common approach: split into up to 3 spaces by cutting along x, y, z.
        
        new_spaces = []
        # Space above the box (in y direction)
        if sh > bh:
            new_spaces.append((sx, sy + bh, sz, sw, sh - bh, sd))
        # Space to the right of the box (in x direction), but only the part not already covered by "above"
        if sw > bw:
            new_spaces.append((sx + bw, sy, sz, sw - bw, sh, sd))
        # Space behind the box (in z direction), but only the part not covered by "above" or "right"
        if sd > bd:
            # The "behind" space should only include the part where x < sx+bw and y < sy+bh
            # Actually, let's use a simpler decomposition:
            # After placing box at (sx, sy, sz) with dims (bw, bh, bd) in space (sx, sy, sz, sw, sh, sd):
            # Region 1: y from sy+bh to sy+sh, x from sx to sx+sw, z from sz to sz+sd -> (sx, sy+bh, sz, sw, sh-bh, sd)
            # Region 2: x from sx+bw to sx+sw, y from sy to sy+bh, z from sz to sz+sd -> (sx+bw, sy, sz, sw-bw, bh, sd)
            # Region 3: z from sz+bd to sz+sd, x from sx to sx+bw, y from sy to sy+bh -> (sx, sy, sz+bd, bw, bh, sd-bd)
            new_spaces.append((sx, sy, sz + bd, bw, bh, sd - bd))
        
        return new_spaces
    
    def can_fit(box_dims, space):
        """Check if box fits in space."""
        bw, bh, bd = box_dims
        _, _, _, sw, sh, sd = space
        return bw <= sw and bh <= sh and bd <= sd
    
    def find_best_position(box_dims, all_spaces):
        """
        Find the best (bin_idx, space_idx) where the box fits.
        Best = smallest resulting empty space volume (best fit).
        Returns (bin_idx, space_idx, space) or None.
        """
        best = None
        best_score = float('inf')
        bw, bh, bd = box_dims
        for bi, spaces in enumerate(all_spaces):
            for si, space in enumerate(spaces):
                if can_fit(box_dims, space):
                    # Calculate the "waste" or resulting fragmentation
                    # Best fit: minimize the volume of the smallest remaining space after placement
                    # Or minimize the sum of volumes of remaining spaces? 
                    # A common metric: minimize the maximum dimension of the remaining space, 
                    # or minimize the volume of the largest remaining space.
                    # Let's use: minimize the volume of the empty space that is "leftover" in a way that 
                    # keeps spaces compact. 
                    # Simple approach: minimize the product of the "gaps" or just use the space volume as a proxy.
                    # Actually, best fit usually means: choose the space where the box fits with least waste.
                    # Waste = space_volume - box_volume, but that's constant for a given space.
                    # Better: after splitting, look at the resulting spaces. Minimize the largest remaining space volume?
                    # Or minimize the sum of squared volumes?
                    # For simplicity, let's minimize the volume of the space (tighter fit = smaller space).
                    space_vol = space[3] * space[4] * space[5]
                    if space_vol < best_score:
                        best_score = space_vol
                        best = (bi, si, space)
        return best
    
    def place_box(box_idx, box_dims, bin_idx, space_idx, space):
        """Place a box in the given bin at the given space. Update empty spaces."""
        sx, sy, sz, sw, sh, sd = space
        # Remove the space
        bins_spaces = bin_empty_spaces[bin_idx]
        bins_spaces.pop(space_idx)
        # Split and add new spaces
        new_spaces = split_space(space, box_dims)
        bins_spaces.extend(new_spaces)
        bin_volumes[bin_idx] += box_dims[0] * box_dims[1] * box_dims[2]
    
    # Main loop: place each box
    for i, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
        
        box_dims = (w, h, d)
        
        # Try to place in existing bins using best fit
        best = find_best_position(box_dims, bin_empty_spaces)
        
        if best is None:
            # Open a new bin
            bin_idx = len(bins)
            bins.append([])
            bin_empty_spaces.append([[0, 0, 0, W, H, D]])
            bin_volumes.append(0)
            space_idx = 0
            space = bin_empty_spaces[bin_idx][0]
            place_box(orig_idx, box_dims, bin_idx, space_idx, space)
            placements[orig_idx] = (bin_idx, 0, 0, 0, w, h, d)
        else:
            bin_idx, space_idx, space = best
            sx, sy, sz, _, _, _ = space
            place_box(orig_idx, box_dims, bin_idx, space_idx, space)
            placements[orig_idx] = (bin_idx, sx, sy, sz, w, h, d)
    
    # Handle any boxes not placed (shouldn't happen if we have enough bins, but just in case)
    for i in range(len(boxes)):
        if placements[i] is None:
            # This box wasn't placed due to time limit; place it in a new bin
            bin_idx = len(bins)
            bins.append([])
            bin_empty_spaces.append([[0, 0, 0, W, H, D]])
            bin_volumes.append(0)
            w, h, d = boxes[i]
            placements[i] = (bin_idx, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
