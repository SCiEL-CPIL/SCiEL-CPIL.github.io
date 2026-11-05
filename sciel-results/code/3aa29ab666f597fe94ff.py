"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # We'll try to place boxes one by one into existing bins or open new ones
    # Each bin is a list of (x, y, z, w, h, d)
    bins = []  # list of lists of placements
    
    def fits_in_bin(box, bin_placements):
        """Check if box fits in a bin without overlap, return (x,y,z) or None."""
        w, h, d = box[1], box[2], box[3]
        
        # Candidate positions:
        # 1. Origin
        # 2. On top of each existing box (z = existing_z + existing_d)
        # 3. Beside each existing box in x (x = existing_x + existing_w)
        # 4. Beside each existing box in y (y = existing_y + existing_h)
        # 5. Combinations: on top of one, beside another in x/y
        
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (ex, ey, ez, ew, eh, ed) in bin_placements:
            candidates.add((ex + ew, ey, ez))
            candidates.add((ex, ey + eh, ez))
            candidates.add((ex, ey, ez + ed))
            candidates.add((ex + ew, ey + eh, ez))
            candidates.add((ex + ew, ey, ez + ed))
            candidates.add((ex, ey + eh, ez + ed))
            candidates.add((ex + ew, ey + eh, ez + ed))
        
        best_pos = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            # Check overlap with all existing boxes
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in bin_placements:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                # Score: prefer lower z, then lower y, then lower x (bottom-fill)
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        
        return best_pos
    
    def bin_volume(bin_placements):
        return sum(w*h*d for (_, _, _, w, h, d) in bin_placements)
    
    placements = []
    
    for idx, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            # Fallback: place remaining boxes in a new bin with simple shelf
            # Actually, we need to ensure all boxes are placed. Let's use a simple
            # approach for remaining boxes.
            # Create a new bin and use shelf packing for the rest
            b = len(bins)
            x = y = z = 0
            row_h = layer_d = 0
            for rem_idx in range(idx, n):
                rw, rh, rd = boxes[rem_idx]
                if x + rw > W:
                    x, y, row_h = 0, y + row_h, 0
                if y + rh > H:
                    y, z, layer_d = 0, z + layer_d, 0
                    x, row_h = 0, 0
                if z + rd > D:
                    b += 1
                    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
                placements.append((b, x, y, z, rw, rh, rd))
                x, row_h, layer_d = x + rw, max(row_h, rh), max(layer_d, rd)
            # Now we need to merge with existing placements
            # This is getting complicated. Let's just return what we have plus these.
            # Actually, let's restructure: build a list of (orig_idx, bin, x, y, z, w, h, d)
            break
        
        box = (orig_idx, w, h, d, vol)
        best_bin = -1
        best_pos = None
        best_score = None
        
        for b_idx, bin_pl in enumerate(bins):
            pos = fits_in_bin(box, bin_pl)
            if pos is not None:
                # Score: prefer bins with more volume already used, then lower position
                bvol = bin_volume(bin_pl)
                score = (-bvol, pos[2], pos[1], pos[0])
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            # Open new bin
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
    
    # If we broke out early due to deadline, we need to handle remaining boxes
    # Let's restructure to avoid the break complexity
    
    # Actually, let me rewrite this more cleanly.
    pass

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    bins = []
    
    def find_position(w, h, d, bin_placements):
        candidates = set()
        candidates.add((0, 0, 0))
        for (ex, ey, ez, ew, eh, ed) in bin_placements:
            candidates.add((ex + ew, ey, ez))
            candidates.add((ex, ey + eh, ez))
            candidates.add((ex, ey, ez + ed))
            candidates.add((ex + ew, ey + eh, ez))
            candidates.add((ex + ew, ey, ez + ed))
            candidates.add((ex, ey + eh, ez + ed))
            candidates.add((ex + ew, ey + eh, ez + ed))
        
        best_pos = None
        best_score = None
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in bin_placements:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        return best_pos
    
    result = [None] * n
    
    for idx, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            # Fallback: simple shelf for remaining
            remaining = indexed_boxes[idx:]
            b = len(bins)
            x = y = z = 0
            row_h = layer_d = 0
            for (oi, rw, rh, rd, rv) in remaining:
                if x + rw > W:
                    x, y, row_h = 0, y + row_h, 0
                if y + rh > H:
                    y, z, layer_d = 0, z + layer_d, 0
                    x, row_h = 0, 0
                if z + rd > D:
                    b += 1
                    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
                result[oi] = (b, x, y, z, rw, rh, rd)
                x, row_h, layer_d = x + rw, max(row_h, rh), max(layer_d, rd)
            break
        
        best_bin = -1
        best_pos = None
        best_score = None
        
        for b_idx, bin_pl in enumerate(bins):
            pos = find_position(w, h, d, bin_pl)
            if pos is not None:
                bvol = sum(ew*eh*ed for (_, _, _, ew, eh, ed) in bin_pl)
                score = (-bvol, pos[2], pos[1], pos[0])
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        result[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
    
    return result
# EVOLVE-BLOCK-END
