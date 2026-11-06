"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pack(boxes, container):
    """
    Place boxes into bins using a Best-Fit Decreasing with 3D shelf heuristic.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Sort boxes by volume descending for BFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # Each bin will have a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists

    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) in the given bin without overlap and within bounds."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def find_position(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin.
        Uses a shelf-based approach: try to place on existing shelves or create new ones.
        Returns (x, y, z) or None if no position found.
        """
        # Collect all existing shelf levels (y positions) and their max heights
        # Also consider z-levels
        # Strategy: iterate over potential y positions (0 and top of each existing shelf)
        # and for each, iterate over potential z positions (0 and top of each existing layer)
        # and find the smallest x that works.
        
        # Build a set of candidate y positions
        y_candidates = {0}
        for (px, py, pz, pw, ph, pd) in bin_placements:
            y_candidates.add(py + ph)
        
        # For each y candidate, find candidate z positions
        best_pos = None
        best_score = None
        
        for yc in sorted(y_candidates):
            if yc + h > H:
                continue
            
            # Find z candidates for this y level
            z_candidates = {0}
            for (px, py, pz, pw, ph, pd) in bin_placements:
                if py == yc and pz + pd <= yc + h:  # boxes at this y level or below that might constrain z
                    # Actually, we need z positions that are at or above existing boxes at this y
                    z_candidates.add(pz + pd)
            
            for zc in sorted(z_candidates):
                if zc + d > D:
                    continue
                
                # Find smallest x that works
                x_candidates = {0}
                for (px, py, pz, pw, ph, pd) in bin_placements:
                    if py == yc and pz == zc:
                        x_candidates.add(px + pw)
                
                for xc in sorted(x_candidates):
                    if xc + w > W:
                        continue
                    if can_place(bin_placements, xc, yc, zc, w, h, d):
                        # Score: prefer lower y, then lower z, then lower x
                        score = (yc, zc, xc)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (xc, yc, zc)
        
        return best_pos

    # Main packing loop
    result = [None] * n  # will hold (bin_id, x, y, z, w, h, d) for each original index
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place all remaining boxes in a new bin each (valid but poor)
            # Actually, let's just place them in a single new bin using the simple shelf method
            break
        
        placed = False
        # Try to place in existing bins, preferring the bin where it fits tightest
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer positions that minimize leftover space
                # Use a simple heuristic: prefer lower (x+y+z) or minimize max coordinate
                fit_score = (y + h, z + d, x + w)
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Create a new bin
            new_bin = []
            x, y, z = 0, 0, 0
            if can_place(new_bin, x, y, z, w, h, d):
                new_bin.append((x, y, z, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, x, y, z, w, h, d)
            else:
                # This shouldn't happen if container is large enough, but handle gracefully
                # Place at origin in a new bin (may overlap if invalid, but assume valid input)
                new_bin.append((0, 0, 0, w, h, d))
                bins.append(new_bin)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # If we hit the deadline, place remaining boxes using a simple fallback
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is not None:
            continue
        # Find any bin where it can be placed, or create a new one
        placed = False
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    return result
# EVOLVE-BLOCK-END
