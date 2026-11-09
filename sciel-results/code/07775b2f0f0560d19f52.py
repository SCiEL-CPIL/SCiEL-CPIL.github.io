"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Binary Space Partitioning (BSP) trees.
    
    Each bin maintains a BSP tree that recursively subdivides the container
    into empty rectangular subspaces. When placing a box, we search for the
    best empty rectangular region that can contain the box, then insert it by
    splitting that region.
    
    We process boxes in order of decreasing volume (largest first) to improve
    packing efficiency, then map back to original order.
    
    Return: list of (bin, x, y, z, w, h, d) for each box in input order.
    """
    W, H, D = container
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, box) for i, box in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -(x[1][0] * x[1][1] * x[1][2]))
    
    # Each bin: list of free rectangular regions (x, y, z, w, h, d)
    bins = []  # list of lists of regions
    
    def split_region(region, box):
        """
        Given a free region and a box to place at the corner (region's x,y,z),
        split the region into up to 3 new free regions around the box.
        The box is placed at (region[0], region[1], region[2]).
        Returns list of new free regions.
        """
        rx, ry, rz, rw, rh, rd = region
        bw, bh, bd = box
        
        new_regions = []
        
        # Region to the right: (rx+bw, ry, rz, rw-bw, rh, rd)
        if rw - bw > 0:
            new_regions.append((rx + bw, ry, rz, rw - bw, rh, rd))
        
        # Region above: (rx, ry+bh, rz, bw, rh-bh, rd)
        if rh - bh > 0:
            new_regions.append((rx, ry + bh, rz, bw, rh - bh, rd))
        
        # Region behind: (rx, ry, rz+bd, bw, bh, rd-bd)
        if rd - bd > 0:
            new_regions.append((rx, ry, rz + bd, bw, bh, rd - bd))
        
        return new_regions
    
    def region_waste(region, box):
        """
        Calculate the 'waste' when placing box in region.
        We prefer placements that minimize the sum of distances from origin
        and maximize the remaining space in a compact way.
        """
        rx, ry, rz, rw, rh, rd = region
        bw, bh, bd = box
        
        # Penalize being far from origin
        dist_penalty = (rx + ry + rz) * 0.001
        
        # Calculate remaining volumes after placement
        # Right: (rw-bw) * rh * rd
        # Above: bw * (rh-bh) * rd
        # Behind: bw * bh * (rd-bd)
        vol_right = max(0, rw - bw) * rh * rd
        vol_above = bw * max(0, rh - bh) * rd
        vol_behind = bw * bh * max(0, rd - bd)
        
        # Prefer placements where remaining space is more "compact" (less fragmented)
        # Use a simple heuristic: minimize the max of the three remaining volumes
        max_vol = max(vol_right, vol_above, vol_behind)
        
        return dist_penalty + max_vol * 0.0001
    
    def find_and_place(bins, box):
        """
        Try to place box (w, h, d) in one of the existing bins.
        Returns (bin_index, x, y, z) if successful, or None.
        """
        w, h, d = box
        
        best_bin = -1
        best_pos = (float('inf'), float('inf'), float('inf'))
        best_waste = float('inf')
        best_region_idx = -1
        
        for bi in range(len(bins)):
            free_regions = bins[bi]
            
            for i, region in enumerate(free_regions):
                rx, ry, rz, rw, rh, rd = region
                if rw >= w and rh >= h and rd >= d:
                    waste = region_waste(region, (w, h, d))
                    if waste < best_waste:
                        best_waste = waste
                        best_bin = bi
                        best_pos = (rx, ry, rz)
                        best_region_idx = i
        
        if best_bin >= 0:
            region = bins[best_bin][best_region_idx]
            
            # Remove the region
            bins[best_bin].pop(best_region_idx)
            
            # Split and add new regions
            new_regions = split_region(region, (w, h, d))
            bins[best_bin].extend(new_regions)
            
            return (best_bin, best_pos[0], best_pos[1], best_pos[2])
        
        return None
    
    # Process boxes in sorted order
    placements = [None] * len(boxes)
    
    for orig_idx, box in indexed_boxes:
        w, h, d = box
        
        # Try to place in existing bins
        result = find_and_place(bins, (w, h, d))
        
        if result is not None:
            bi, x, y, z = result
            placements[orig_idx] = (bi, x, y, z, w, h, d)
        else:
            # Open a new bin
            bi = len(bins)
            bins.append([(0, 0, 0, W, H, D)])
            
            # Place at origin
            x, y, z = 0, 0, 0
            
            # Split the initial region
            new_regions = split_region((0, 0, 0, W, H, D), (w, h, d))
            bins[bi] = new_regions
            
            placements[orig_idx] = (bi, x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
