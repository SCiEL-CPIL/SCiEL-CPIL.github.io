"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    Place boxes into bins using a multi-strategy 3D bin packing heuristic.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
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

    def find_position_corner(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin using corner-based placement.
        Generates candidate positions from the corners of existing boxes.
        Returns (x, y, z) or None if no position found.
        """
        # Candidate positions: origin + all "upper" corners of existing boxes
        candidates = {(0, 0, 0)}
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # The corner that extends the box in each dimension
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        best_pos = None
        best_score = None
        
        for (xc, yc, zc) in candidates:
            if xc + w > W or yc + h > H or zc + d > D:
                continue
            if can_place(bin_placements, xc, yc, zc, w, h, d):
                # Score: prefer positions that minimize the resulting bounding box extent
                # Use the maximum of (x+w, y+h, z+d) to minimize overall spread
                score = max(xc + w, yc + h, zc + d)
                # Tiebreak: prefer lower sum of coordinates for compactness
                tiebreak = xc + yc + zc
                combined = (score, tiebreak)
                if best_score is None or combined < best_score:
                    best_score = combined
                    best_pos = (xc, yc, zc)
        
        return best_pos

    def find_position_shelf(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin using shelf-based placement.
        Returns (x, y, z) or None if no position found.
        """
        # Collect all existing shelf levels (y positions)
        shelf_levels = set([0])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            shelf_levels.add(py + ph)
        
        best_pos = None
        best_score = None
        
        for y_level in sorted(shelf_levels):
            if y_level + h > H:
                continue
            # Find the best x position on this shelf
            x_candidates = set([0])
            for (px, py, pz, pw, ph, pd) in bin_placements:
                if py == y_level:
                    x_candidates.add(px + pw)
            
            for x in sorted(x_candidates):
                if x + w > W:
                    continue
                # Find the best z position
                z_candidates = set([0])
                for (px, py, pz, pw, ph, pd) in bin_placements:
                    if py == y_level and px <= x and px + pw >= x + w:
                        z_candidates.add(pz + pd)
                
                for z in sorted(z_candidates):
                    if z + d > D:
                        continue
                    if can_place(bin_placements, x, y_level, z, w, h, d):
                        # Same scoring as corner: minimize bounding box extent
                        score = max(x + w, y_level + h, z + d)
                        tiebreak = x + y_level + z
                        combined = (score, tiebreak)
                        if best_score is None or combined < best_score:
                            best_score = combined
                            best_pos = (x, y_level, z)
                        break
        
        return best_pos

    def find_position(bin_placements, w, h, d):
        """
        Try multiple placement strategies and return the best position.
        """
        # Try corner-based
        pos_corner = find_position_corner(bin_placements, w, h, d)
        if pos_corner is not None:
            return pos_corner
        
        # Try shelf-based
        pos_shelf = find_position_shelf(bin_placements, w, h, d)
        if pos_shelf is not None:
            return pos_shelf
        
        return None

    # Main packing loop
    result = [None] * n  # will hold (bin_id, x, y, z, w, h, d) for each original index
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try to place in existing bins, preferring the bin where it fits tightest
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer bins where the box fits most tightly
                # Use the product of remaining space as a proxy for wasted volume
                remaining_x = W - (x + w)
                remaining_y = H - (y + h)
                remaining_z = D - (z + d)
                # Smaller remaining space = tighter fit
                fit_score = remaining_x * remaining_y * remaining_z
                # Tiebreak: prefer lower bounding box extent
                bbox_extent = max(x + w, y + h, z + d)
                combined = (fit_score, bbox_extent)
                if best_fit_score is None or combined < best_fit_score:
                    best_fit_score = combined
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
