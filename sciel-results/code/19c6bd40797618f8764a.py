"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # 1. Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # bins is a list of lists. Each inner list contains (x, y, z, w, h, d, orig_idx)
    bins = []
    bin_vols = [] 
    bin_grids = []
    bin_bboxes = [] # (max_x, max_y, max_z)

    def fits_in_bin(bin_grid, x, y, z, w, h, d):
        """Check if box fits at position (x,y,z) without overlap and within bounds using grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any part of the box overlaps with occupied grid cells
        if np.any(bin_grid[x:x+w, y:y+h, z:z+d]):
            return False
        return True

    def find_position(bin_items, bin_grid, w, h, d):
        """
        Find the best position for a box in a bin.
        Heuristic: Bottom-Left-Back (BLB). Minimize (x + y + z).
        Tie-breaker: Minimize (x * H * D + y * D + z) to prioritize lower x, then y, then z.
        """
        if not bin_items:
            return (0, 0, 0)
        
        candidates = [(0, 0, 0)]
        
        # Generate candidates based on faces and corners of existing boxes
        # We want to place boxes as close to the origin as possible.
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            # Faces
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            
            # Corners (combinations of face offsets)
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
            
            # Also try positions aligned with the other axes of the current box
            # e.g., if there is a box at (0,0,0) with size (10, 5, 5),
            # and we are placing a box of size (2, 2, 2),
            # (10, 0, 0) is a candidate. (0, 5, 0) is a candidate.
            # The logic above covers these.

        best_pos = None
        best_score = float('inf')
        
        seen = set()
        for cx, cy, cz in candidates:
            if (cx, cy, cz) in seen:
                continue
            seen.add((cx, cy, cz))
            
            if fits_in_bin(bin_grid, cx, cy, cz, w, h, d):
                # Score: Bottom-Left-Back
                # Primary: sum of coordinates
                # Secondary: lexicographic order (x, then y, then z) to ensure consistency
                score = cx + cy + cz
                # To break ties strictly by BLB priority:
                # We can use a tuple (cx + cy + cz, cx, cy, cz)
                current_tuple = (score, cx, cy, cz)
                
                if best_pos is None:
                    best_pos = (cx, cy, cz)
                    best_score = current_tuple
                else:
                    # Compare tuples
                    if current_tuple < best_score:
                        best_score = current_tuple
                        best_pos = (cx, cy, cz)
                        
        return best_pos

    deadline = time.time() + 145.0 # Safety margin
    
    for i, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback: place all remaining boxes in new bins
            for rem_i, rw, rh, rd, rv in indexed_boxes[i:]:
                bins.append([(0, 0, 0, rw, rh, rd, rem_i)])
                bin_vols.append(rw*rh*rd)
                # Create grid for new bin
                grid = np.zeros((W, H, D), dtype=bool)
                grid[0:rw, 0:rh, 0:rd] = True
                bin_grids.append(grid)
                bin_bboxes.append((rw, rh, rd))
            break
            
        best_bin_idx = -1
        best_pos = None
        best_bin_score = -1
        
        # Try to fit in existing bins
        for b_idx, bin_items in enumerate(bins):
            bin_grid = bin_grids[b_idx]
            pos = find_position(bin_items, bin_grid, w, h, d)
            if pos is not None:
                # Score for bin selection: Prefer bins that are already fuller
                # to minimize the number of bins used (B)
                current_bin_vol = bin_vols[b_idx]
                if current_bin_vol > best_bin_score:
                    best_bin_score = current_bin_vol
                    best_bin_idx = b_idx
                    best_pos = pos
                    
        if best_bin_idx == -1:
            # No existing bin works, start a new one
            bins.append([(0, 0, 0, w, h, d, i)])
            bin_vols.append(w*h*d)
            grid = np.zeros((W, H, D), dtype=bool)
            grid[0:w, 0:h, 0:d] = True
            bin_grids.append(grid)
            bin_bboxes.append((w, h, d))
        else:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, i))
            bin_vols[best_bin_idx] += w*h*d
            bin_grids[best_bin_idx][best_pos[0]:best_pos[0]+w, best_pos[1]:best_pos[1]+h, best_pos[2]:best_pos[2]+d] = True
            
            # Update bounding box (not strictly used in BLB, but good for potential future heuristics)
            ox, oy, oz, ow, oh, od, _ = bins[best_bin_idx][-1]
            old_bbox = bin_bboxes[best_bin_idx]
            new_max_x = max(old_bbox[0], ox + ow)
            new_max_y = max(old_bbox[1], oy + oh)
            new_max_z = max(old_bbox[2], oz + od)
            bin_bboxes[best_bin_idx] = (new_max_x, new_max_y, new_max_z)

    # Construct final placements indexed by original box index
    placements = [None] * n
    
    for b_idx, bin_items in enumerate(bins):
        for item in bin_items:
            x, y, z, w, h, d, orig_idx = item
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
