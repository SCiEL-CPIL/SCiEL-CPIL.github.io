import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    Improved 3D bin packing using Best-Fit Decreasing with systematic candidate generation.
    
    Strategy:
    1. Sort boxes by volume descending, with tie-breaking by max dimension.
    2. For each box, evaluate all existing bins using systematic candidate points.
    3. Score positions by bounding box expansion and prefer fuller bins.
    4. If no existing bin fits, start a new bin.
    5. Uses deadline guard for safety.
    
    Returns:
    List of placements: (bin_id, x, y, z, w, h, d)
    """
    W, H, D = container
    container_vol = W * H * D
    
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Safety margin before 120s limit
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: (x[1][0]*x[1][1]*x[1][2], max(x[1])), reverse=True)
    
    bins = []  # List of lists of (x,y,z,w,h,d)
    bin_volumes = []  # Track total volume per bin for quick access
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True

    def compute_bounding_vol(bin_boxes, x, y, z, w, h, d):
        """Compute the bounding box volume after placing this box."""
        if not bin_boxes:
            return w * h * d
        
        max_x = max(bx + bw for bx, by, bz, bw, bh, bd in bin_boxes)
        max_y = max(by + bh for bx, by, bz, bw, bh, bd in bin_boxes)
        max_z = max(bz + bd for bx, by, bz, bw, bh, bd in bin_boxes)
        
        new_max_x = max(max_x, x + w)
        new_max_y = max(max_y, y + h)
        new_max_z = max(max_z, z + d)
        
        return new_max_x * new_max_y * new_max_z

    def get_current_bounding_vol(bin_boxes):
        """Get the current bounding box volume of a bin."""
        if not bin_boxes:
            return 0
        max_x = max(bx + bw for bx, by, bz, bw, bh, bd in bin_boxes)
        max_y = max(by + bh for bx, by, bz, bw, bh, bd in bin_boxes)
        max_z = max(bz + bd for bx, by, bz, bw, bh, bd in bin_boxes)
        return max_x * max_y * max_z

    def find_best_position(bin_boxes, w, h, d):
        """Find the position that minimizes bounding box expansion for a given bin."""
        if not bin_boxes:
            if can_place(bin_boxes, 0, 0, 0, w, h, d):
                return (0, 0, 0)
            return None

        candidates = set()
        candidates.add((0, 0, 0))
        
        # Generate candidate points from existing boxes
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # All combinations of offset positions
            for dx in [0, bw]:
                for dy in [0, bh]:
                    for dz in [0, bd]:
                        candidates.add((bx + dx, by + dy, bz + dz))
            
            # Floor-aligned candidates
            candidates.add((bx + bw, by, 0))
            candidates.add((bx, by + bh, 0))
            candidates.add((0, by, bz + bd))
            candidates.add((bx, 0, bz + bd))
            candidates.add((bx + bw, 0, 0))
            candidates.add((0, by + bh, 0))
            candidates.add((0, 0, bz + bd))
        
        # Cross-axis combinations
        n = len(bin_boxes)
        for i in range(n):
            bx1, by1, bz1, bw1, bh1, bd1 = bin_boxes[i]
            for j in range(i + 1, n):
                bx2, by2, bz2, bw2, bh2, bd2 = bin_boxes[j]
                # Mix x from one, y from another
                candidates.add((bx1 + bw1, by2, bz1))
                candidates.add((bx2, by1 + bh1, bz2))
                candidates.add((bx1 + bw1, by2, bz2 + bd2))
                candidates.add((bx2 + bw2, by1, bz1))
        
        # Filter valid candidates
        valid_candidates = []
        for x, y, z in candidates:
            if x < 0 or y < 0 or z < 0:
                continue
            if x + w > W or y + h > H or z + d > D:
                continue
            if can_place(bin_boxes, x, y, z, w, h, d):
                # Compute bounding box expansion
                current_bv = get_current_bounding_vol(bin_boxes)
                new_bv = compute_bounding_vol(bin_boxes, x, y, z, w, h, d)
                expansion = new_bv - current_bv
                # Also consider coordinate sum for compactness
                coord_sum = x + y + z
                valid_candidates.append((expansion, coord_sum, x, y, z, (x, y, z)))
        
        if not valid_candidates:
            return None
        
        # Sort by expansion first, then by coordinate sum
        valid_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        return valid_candidates[0][5]

    # Store results in original order
    results = [None] * len(boxes)
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() - start_time > timeout:
            # Fast fallback: place remaining in new bins
            for remaining_idx, (rw, rh, rd) in indexed_boxes:
                if results[remaining_idx] is None:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    bin_volumes.append(rw * rh * rd)
                    results[remaining_idx] = (b_idx, 0, 0, 0, rw, rh, rd)
            break

        placed = False
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        # Try existing bins and find the one with minimum score
        for b_idx in range(len(bins)):
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: bounding box expansion - bonus for higher fill ratio
                current_bv = get_current_bounding_vol(bins[b_idx])
                new_bv = compute_bounding_vol(bins[b_idx], x, y, z, w, h, d)
                expansion = new_bv - current_bv
                
                # Fill ratio bonus: prefer bins that are already fuller
                fill_ratio = bin_volumes[b_idx] / container_vol
                # Combined score: minimize expansion, but give bonus for fuller bins
                combined_score = expansion - 100 * fill_ratio
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            bin_volumes[best_bin] += w * h * d
            results[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Start new bin
            b_idx = len(bins)
            x, y, z = 0, 0, 0
            bins.append([(x, y, z, w, h, d)])
            bin_volumes.append(w * h * d)
            results[idx] = (b_idx, x, y, z, w, h, d)

    return results

# EVOLVE-BLOCK-END
