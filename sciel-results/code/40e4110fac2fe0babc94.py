"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Binary Space Partitioning (BSP) trees.
    
    Improvements:
    1. Sort boxes by volume (descending) before packing to improve bin utilization.
    2. Use a 'Best Fit' heuristic: among all valid free regions in a bin,
       choose the one that minimizes the sum of coordinates (x+y+z),
       with a tie-breaker favoring the region with the smallest volume (tightest fit).
    3. Restore original box order in the final output.
    """
    W, H, D = container
    container_vol = W * H * D if W > 0 and H > 0 and D > 0 else 1
    
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
    
    def find_and_place(bins, box):
        """
        Try to place box (w, h, d) in one of the existing bins.
        Returns (bin_index, x, y, z) if successful, or None.
        Uses Best Fit heuristic: minimize (x+y+z), then minimize region volume.
        """
        w, h, d = box
        
        for bi in range(len(bins)):
            free_regions = bins[bi]
            
            best_idx = -1
            best_score = (float('inf'), float('inf')) # (coord_sum, region_vol)
            
            for i, region in enumerate(free_regions):
                rx, ry, rz, rw, rh, rd = region
                if rw >= w and rh >= h and rd >= d:
                    coord_sum = rx + ry + rz
                    region_vol = rw * rh * rd
                    score = (coord_sum, region_vol)
                    if score < best_score:
                        best_score = score
                        best_idx = i
            
            if best_idx >= 0:
                region = free_regions[best_idx]
                rx, ry, rz, rw, rh, rd = region
                
                # Remove the region
                free_regions.pop(best_idx)
                
                # Split and add new regions
                new_regions = split_region(region, (w, h, d))
                free_regions.extend(new_regions)
                
                return (bi, rx, ry, rz)
        
        return None
    
    # Create a list of (volume, original_index, box_dims)
    indexed_boxes = []
    for i, box in enumerate(boxes):
        w, h, d = box
        vol = w * h * d
        indexed_boxes.append((vol, i, box))
    
    # Sort by volume descending to pack larger items first
    indexed_boxes.sort(key=lambda x: x[0], reverse=True)
    
    # Prepare storage for results, mapping original index to placement
    placements = [None] * len(boxes)
    
    for vol, orig_idx, box in indexed_boxes:
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
