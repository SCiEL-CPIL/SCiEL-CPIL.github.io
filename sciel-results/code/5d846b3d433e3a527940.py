"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # 1. Sort boxes by volume descending (Best Fit Decreasing)
    # We keep the original index to map back later
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # bins is a list of lists. Each inner list contains (x, y, z, w, h, d, orig_idx)
    bins = []
    bin_vols = [] # Track volume in each bin for faster scoring
    bin_grids = [] # Track occupancy grids for O(1) overlap checks

    def fits_in_bin(bin_grid, x, y, z, w, h, d):
        """Check if box fits at position (x,y,z) without overlap and within bounds using grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        if np.any(bin_grid[x:x+w, y:y+h, z:z+d]):
            return False
        return True

    def find_position(bin_items, bin_grid, w, h, d):
        """
        Find the best position for a box in a bin.
        """
        if not bin_items:
            return (0, 0, 0)
        
        candidates = [(0, 0, 0)]
        
        # Generate candidates based on faces of existing boxes
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            # Right face
            candidates.append((ox + ow, oy, oz))
            # Top face
            candidates.append((ox, oy + oh, oz))
            # Front face
            candidates.append((ox, oy, oz + od))
            
            # Add corner combinations to fill gaps better
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))

        best_pos = None
        best_score = float('inf')
        
        seen = set()
        for cx, cy, cz in candidates:
            if (cx, cy, cz) in seen:
                continue
            seen.add((cx, cy, cz))
            
            if fits_in_bin(bin_grid, cx, cy, cz, w, h, d):
                # Score: prefer positions closer to origin (0,0,0)
                # Also penalize height (z) more to encourage flat packing?
                # Standard "bottom-left-front" heuristic: x+y+z is common.
                # Let's try x + y + 0.5*z to prioritize horizontal spread over vertical stacking?
                # Or just x+y+z is fine.
                score = cx + cy + cz
                if score < best_score:
                    best_score = score
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
        else:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, i))
            bin_vols[best_bin_idx] += w*h*d
            bin_grids[best_bin_idx][best_pos[0]:best_pos[0]+w, best_pos[1]:best_pos[1]+h, best_pos[2]:best_pos[2]+d] = True

    # Construct final placements indexed by original box index
    # The evaluator expects placements[i] to correspond to boxes[i]
    placements = [None] * n
    
    for b_idx, bin_items in enumerate(bins):
        for item in bin_items:
            x, y, z, w, h, d, orig_idx = item
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
            
    # Verify all boxes are placed. If not, it's an error in our logic.
    # However, to be safe against any edge cases where a box might be missed,
    # we can check for None. If None, we place it in a new bin at (0,0,0) as a last resort,
    # though this should be invalid if it doesn't fit, but better than crashing.
    # Given the logic, all boxes are processed.
    
    return placements
# EVOLVE-BLOCK-END
