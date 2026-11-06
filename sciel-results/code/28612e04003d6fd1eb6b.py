"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def pack(boxes, container):
    """
    Improved 3D bin packing: BFD with optimized placement + local search refinement.
    
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Sort boxes by volume descending, with tie-breaking by max dimension
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[1], -max(x[2], x[3], x[4])))
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s kill
    
    # Data structures
    bins = []  # list of lists of (x, y, z, w, h, d)
    placement = {}  # original index -> (bin_idx, x, y, z)
    
    def can_place_fast(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in bin without overlap and within bounds."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Quick reject: check if boxes overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_best_spot(bin_boxes, w, h, d, bin_idx=0):
        """
        Find the best (x, y, z) position for a box in a given bin.
        Uses candidate positions derived from box faces with pruning.
        Returns (x, y, z) or None if no valid position found.
        """
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Build candidate coordinate sets from box edges
        x_vals = [0]
        y_vals = [0]
        z_vals = [0]
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                x_vals.append(px + pw)
            if py + ph <= H:
                y_vals.append(py + ph)
            if pz + pd <= D:
                z_vals.append(pz + pd)
        
        # Deduplicate and sort
        x_vals = sorted(set(x_vals))
        y_vals = sorted(set(y_vals))
        z_vals = sorted(set(z_vals))
        
        # Filter to valid starting positions
        x_vals = [x for x in x_vals if x + w <= W]
        y_vals = [y for y in y_vals if y + h <= H]
        z_vals = [z for z in z_vals if z + d <= D]
        
        if not x_vals or not y_vals or not z_vals:
            return None
        
        # Try positions in order of compactness (minimize x+y+z)
        # Use early termination: stop when we find a position with cost 0
        best_pos = None
        best_cost = float('inf')
        
        # Generate candidate positions sorted by cost
        candidates = []
        for z in z_vals:
            for y in y_vals:
                for x in x_vals:
                    candidates.append((x + y + z, x, y, z))
        
        candidates.sort()
        
        for cost, x, y, z in candidates:
            if cost >= best_cost:
                break  # No better position possible
            if can_place_fast(bin_boxes, x, y, z, w, h, d):
                best_cost = cost
                best_pos = (x, y, z)
                if cost == 0:
                    break  # Can't do better than origin
        
        return best_pos
    
    def compute_bin_fill(bin_boxes):
        """Compute total volume of boxes in a bin."""
        return sum(w * h * d for (_, _, _, w, h, d) in bin_boxes)
    
    def get_score(num_bins, last_bin_fill):
        """Compute the score: B + 0.1*(1 - last_bin_fill_fraction)."""
        if num_bins == 0:
            return 0.0
        last_frac = last_bin_fill / container_vol if container_vol > 0 else 0.0
        return num_bins + 0.1 * (1.0 - last_frac)
    
    def build_result(placement, boxes):
        """Build result in original order."""
        result = []
        for i in range(len(boxes)):
            b_idx, x, y, z = placement[i]
            w_i, h_i, d_i = boxes[i]
            result.append((b_idx, x, y, z, w_i, h_i, d_i))
        return result
    
    def validate_placement(bins, boxes):
        """Validate that no boxes overlap within each bin and all are within bounds."""
        for b_idx, bin_boxes in enumerate(bins):
            for i, (x1, y1, z1, w1, h1, d1) in enumerate(bin_boxes):
                if x1 + w1 > W or y1 + h1 > H or z1 + d1 > D:
                    return False
                for j in range(i + 1, len(bin_boxes)):
                    x2, y2, z2, w2, h2, d2 = bin_boxes[j]
                    if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                            y1 + h1 <= y2 or y2 + h2 <= y1 or
                            z1 + d1 <= z2 or z2 + d2 <= z1):
                        return False
        return True
    
    # Phase 1: Initial packing with BFD
    for idx, vol, w, h, d in indexed_boxes:
        # Time check
        if time.time() > deadline - 30.0:
            # Fallback: place remaining boxes quickly
            for idx2, vol2, w2, h2, d2 in indexed_boxes:
                if idx2 in placement:
                    continue
                placed = False
                for b_idx in range(len(bins)):
                    if can_place_fast(bins[b_idx], 0, 0, 0, w2, h2, d2):
                        bins[b_idx].append((0, 0, 0, w2, h2, d2))
                        placement[idx2] = (b_idx, 0, 0, 0)
                        placed = True
                        break
                if not placed:
                    for b_idx in range(len(bins)):
                        spot = find_best_spot(bins[b_idx], w2, h2, d2)
                        if spot is not None:
                            x, y, z = spot
                            bins[b_idx].append((x, y, z, w2, h2, d2))
                            placement[idx2] = (b_idx, x, y, z)
                            placed = True
                            break
                if not placed:
                    bins.append([(0, 0, 0, w2, h2, d2)])
                    placement[idx2] = (len(bins) - 1, 0, 0, 0)
            
            return build_result(placement, boxes)
        
        # Try to place in existing bins using best-fit
        best_bin = -1
        best_x, best_y, best_z = 0, 0, 0
        best_cost = float('inf')
        
        for b_idx in range(len(bins)):
            spot = find_best_spot(bins[b_idx], w, h, d)
            if spot is not None:
                x, y, z = spot
                cost = x + y + z
                if cost < best_cost:
                    best_cost = cost
                    best_bin = b_idx
                    best_x, best_y, best_z = x, y, z
        
        if best_bin == -1:
            # Create new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_x, best_y, best_z = 0, 0, 0
        
        # Place the box
        bins[best_bin].append((best_x, best_y, best_z, w, h, d))
        placement[idx] = (best_bin, best_x, best_y, best_z)
    
    # Phase 2: Local search to reduce bin count
    # Try to move boxes from the last bin to other bins to eliminate it
    if time.time() < deadline - 10.0:
        current_score = get_score(len(bins), compute_bin_fill(bins[-1]) if bins else 0)
        
        # Try to eliminate bins from the end
        for _ in range(50):  # Limit iterations
            if time.time() > deadline - 10.0:
                break
            
            if len(bins) <= 1:
                break
            
            # Try to move all boxes from last bin to other bins
            last_bin_idx = len(bins) - 1
            last_boxes = list(bins[last_bin_idx])
            if not last_boxes:
                bins.pop()
                continue
            
            # Clear the last bin
            bins[last_bin_idx] = []
            
            # Try to place each box from last bin into other bins
            success = True
            for box in last_boxes:
                x, y, z, w, h, d = box
                placed = False
                for b_idx in range(len(bins) - 1):
                    spot = find_best_spot(bins[b_idx], w, h, d)
                    if spot is not None:
                        bx, by, bz = spot
                        bins[b_idx].append((bx, by, bz, w, h, d))
                        # Update placement for this box
                        # Find original index for this box
                        for idx2, vol2, w2, h2, d2 in indexed_boxes:
                            if placement[idx2] == (last_bin_idx, x, y, z):
                                placement[idx2] = (b_idx, bx, by, bz)
                                break
                        placed = True
                        break
                
                if not placed:
                    # Failed to place, restore
                    success = False
                    break
            
            if success:
                # Remove the empty last bin
                bins.pop()
                # Update all placements that referenced the old last bin
                for idx2 in placement:
                    if placement[idx2][0] == last_bin_idx:
                        # This shouldn't happen if success, but just in case
                        pass
            else:
                # Restore the last bin
                bins[last_bin_idx] = last_boxes
    
    # Final validation and result building
    if not validate_placement(bins, boxes):
        # Fallback: simple sequential packing
        bins = []
        placement = {}
        for i, (w, h, d) in enumerate(boxes):
            placed = False
            for b_idx in range(len(bins)):
                if can_place_fast(bins[b_idx], 0, 0, 0, w, h, d):
                    bins[b_idx].append((0, 0, 0, w, h, d))
                    placement[i] = (b_idx, 0, 0, 0)
                    placed = True
                    break
            if not placed:
                bins.append([(0, 0, 0, w, h, d)])
                placement[i] = (len(bins) - 1, 0, 0, 0)
    
    return build_result(placement, boxes)
# EVOLVE-BLOCK-END
