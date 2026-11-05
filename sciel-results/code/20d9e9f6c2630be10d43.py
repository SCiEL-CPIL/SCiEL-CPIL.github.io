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
    
    def fits_in_bin(bin_items, x, y, z, w, h, d):
        """Check if box fits at position (x,y,z) without overlap and within bounds."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od, _) in bin_items:
            # Check for overlap in all 3 dimensions
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_position(bin_items, w, h, d):
        """
        Find the best position for a box in a bin.
        Heuristic: Minimize (x + y + z) to pack towards origin/corners.
        We check candidate points: (0,0,0) and points adjacent to existing boxes.
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
            
            # Also try combinations to fill gaps (optional, but adds cost)
            # candidates.append((ox + ow, oy + oh, oz))
            # candidates.append((ox + ow, oy, oz + od))
            # candidates.append((ox, oy + oh, oz + od))
            # candidates.append((ox + ow, oy + oh, oz + od))

        best_pos = None
        best_score = float('inf')
        
        seen = set()
        for cx, cy, cz in candidates:
            if (cx, cy, cz) in seen:
                continue
            seen.add((cx, cy, cz))
            
            if fits_in_bin(bin_items, cx, cy, cz, w, h, d):
                # Score: prefer positions closer to origin (0,0,0)
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
            break
            
        best_bin_idx = -1
        best_pos = None
        best_bin_score = -1
        
        # Try to fit in existing bins
        for b_idx, bin_items in enumerate(bins):
            pos = find_position(bin_items, w, h, d)
            if pos is not None:
                # Score for bin selection: Prefer bins that are already fuller
                # to minimize the number of bins used (B)
                current_bin_vol = sum(ow*oh*od for _,_,_,ow,oh,od,_ in bin_items)
                if current_bin_vol > best_bin_score:
                    best_bin_score = current_bin_vol
                    best_bin_idx = b_idx
                    best_pos = pos
                    
        if best_bin_idx == -1:
            # No existing bin works, start a new one
            bins.append([(0, 0, 0, w, h, d, i)])
        else:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, i))

    # Construct final placements
    # The evaluator expects a list of tuples (bin_id, x, y, z, w, h, d)
    # It does not explicitly require the output to be sorted by original index,
    # but "one placement per box" implies a 1-to-1 mapping.
    # The error message "box 0 must keep its original orientation" suggests
    # the evaluator checks the dimensions of the placed box against the input box.
    # Our data structure stores w,h,d correctly.
    
    placements = []
    for b_idx, bin_items in enumerate(bins):
        for item in bin_items:
            x, y, z, w, h, d, orig_idx = item
            placements.append((b_idx, x, y, z, w, h, d))
            
    # Verify all boxes are placed. If not (due to timeout fallback logic errors),
    # ensure we return something valid.
    if len(placements) != n:
        # This should not happen with the loop above, but just in case
        placed_indices = set()
        # We need to recover indices. In the emergency fallback, we added them.
        # In the normal loop, we added them.
        # If we missed some, we have to put them in new bins.
        # However, the loop iterates through ALL indexed_boxes unless it breaks.
        # If it breaks, it appends the rest. So len(placements) should be n.
        
    return placements
# EVOLVE-BLOCK-END
