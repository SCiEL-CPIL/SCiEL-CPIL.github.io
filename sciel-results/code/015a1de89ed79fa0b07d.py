import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by largest dimension descending
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[4], -max(x[1], x[2], x[3])))
    
    # We'll maintain bins as lists of (x, y, z, w, h, d)
    bins = []  # list of lists of placed boxes
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) in the given bin without overlap."""
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Check overlap in all three dimensions
            if (x < bx + bw and x + w > bx and
                y < by + bh and y + h > by and
                z < bz + bd and z + d > bz):
                return False
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in the bin.
        We use a simple heuristic: try to place at the lowest z, then lowest y, then lowest x.
        We generate candidate positions based on existing boxes."""
        
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Generate candidate x, y, z positions
        # x candidates: 0, and right edges of existing boxes
        # y candidates: 0, and top edges of existing boxes
        # z candidates: 0, and front edges of existing boxes
        
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            x_candidates.append(bx + bw)
            y_candidates.append(by + bh)
            z_candidates.append(bz + bd)
        
        # Also add W, H, D as upper bounds (though they won't be valid start positions)
        # Sort and deduplicate
        x_candidates = sorted(set(x for x in x_candidates if x + w <= W))
        y_candidates = sorted(set(y for y in y_candidates if y + h <= H))
        z_candidates = sorted(set(z for z in z_candidates if z + d <= D))
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None
        
        # Try all combinations, preferring lowest z, then lowest y, then lowest x
        best_pos = None
        best_score = None
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def place_box(bin_idx, x, y, z, w, h, d):
        """Place a box in the specified bin."""
        bins[bin_idx].append((x, y, z, w, h, d))
    
    def get_bin_volume(bin_boxes):
        """Calculate the total volume of boxes in a bin."""
        return sum(bw * bh * bd for (_, _, _, bw, bh, bd) in bin_boxes)
    
    # Greedy placement: for each box (sorted by volume), try to place in existing bins
    # Prefer bins that are already open (lower index first), and within a bin, find best position
    # If no existing bin works, open a new bin
    
    placements = [None] * n  # Will store (bin_idx, x, y, z, w, h, d)
    
    for idx, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
        
        placed = False
        
        # Try existing bins first, in order of their current volume (fill fuller bins first)
        # Actually, let's try all existing bins and pick the one where the box fits with best position
        best_bin = -1
        best_pos = None
        best_pos_score = None
        
        for b_idx in range(len(bins)):
            pos = find_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: prefer bins that are already more filled (to consolidate)
                # And prefer positions with lower z, y, x
                bin_vol = get_bin_volume(bins[b_idx])
                pos_score = (-bin_vol, z, y, x)
                if best_pos_score is None or pos_score < best_pos_score:
                    best_pos_score = pos_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            place_box(best_bin, x, y, z, w, h, d)
            placements[orig_idx] = (best_bin, x, y, z, w, h, d)
            placed = True
        else:
            # Open a new bin
            new_bin_idx = len(bins)
            bins.append([])
            if w <= W and h <= H and d <= D:
                place_box(new_bin_idx, 0, 0, 0, w, h, d)
                placements[orig_idx] = (new_bin_idx, 0, 0, 0, w, h, d)
                placed = True
        
        if not placed:
            # This shouldn't happen if boxes fit in the container, but handle gracefully
            # Place at origin in a new bin as fallback
            new_bin_idx = len(bins)
            bins.append([])
            place_box(new_bin_idx, 0, 0, 0, w, h, d)
            placements[orig_idx] = (new_bin_idx, 0, 0, 0, w, h, d)
    
    # If we timed out, we might have some placements as None
    # Fill in any remaining with a simple shelf heuristic in a new bin
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            new_bin_idx = len(bins)
            bins.append([])
            place_box(new_bin_idx, 0, 0, 0, w, h, d)
            placements[i] = (new_bin_idx, 0, 0, 0, w, h, d)
    
    return placements
