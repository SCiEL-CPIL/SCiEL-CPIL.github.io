"""3D Bin Packing: BFD with adjacency-aware positioning, fragmentation-aware bin selection, and consolidation polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def tightness_score(bin_boxes, x, y, z, w, h, d):
        """
        Score how tightly a box fits. Lower is better.
        Penalizes the 'envelope' size (max extent) and rewards being in the corner.
        """
        if not bin_boxes:
            # For empty bin, just use corner preference
            return 0.5 * x + 0.3 * y + 0.2 * z
        
        # Compute the bounding box of all existing boxes
        max_x = max(px + pw for (px, py, pz, pw, ph, pd) in bin_boxes)
        max_y = max(py + ph for (px, py, pz, pw, ph, pd) in bin_boxes)
        max_z = max(pz + pd for (px, py, pz, pw, ph, pd) in bin_boxes)
        
        # The new box extends the envelope if it goes beyond current bounds
        new_max_x = max(max_x, x + w)
        new_max_y = max(max_y, y + h)
        new_max_z = max(max_z, z + d)
        
        # Penalize growth of the envelope
        envelope_growth = (new_max_x - max_x) + (new_max_y - max_y) + (new_max_z - max_z)
        
        # Also reward being in the corner (lower coords)
        corner_pref = 0.3 * x + 0.3 * y + 0.4 * z
        
        # Combine: primary is envelope growth, secondary is corner preference
        return envelope_growth + 0.5 * corner_pref
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a bin using candidate points."""
        xs = [0]
        ys = [0]
        zs = [0]
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.append(px + pw)
            if py + ph <= H:
                ys.append(py + ph)
            if pz + pd <= D:
                zs.append(pz + pd)
        
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        zs = sorted(set(zs))
        
        best_score = float('inf')
        best_pos = None
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        score = tightness_score(bin_boxes, x, y, z, w, h, d)
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_fill_fraction(bin_boxes):
        """Compute the volume fraction of a bin that is occupied."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / (W * H * D)
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Bin selection: prefer bins that are already well-filled
                # and where the box fits in the tightest spot
                fill = bin_fill_fraction(bin_boxes)
                # The tightness score already captures position quality
                # We want to consolidate: prefer higher fill
                # Use a combined score that rewards both tightness and consolidation
                pos_score = tightness_score(bin_boxes, x, y, z, w, h, d)
                # Weight: 60% position tightness, 40% bin fill (consolidation)
                score = 0.6 * pos_score + 0.4 * (1.0 - fill) * max(W, H, D)
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            for i, (w2, h2, d2) in enumerate(boxes[idx:]):
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[idx + i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Consolidation polish: try to move boxes from later bins into earlier bins
    # Only if we have time remaining
    if time.time() < deadline - 15.0 and len(bins) > 1:
        # Track which original box index is at each position for efficient updates
        # Rebuild: for each bin, store (box_data, original_index)
        bin_box_indices = [[] for _ in bins]
        for orig_idx, (bi, bx, by, bz, bw, bh, bd) in enumerate(placements):
            if bi < len(bins):
                # Find the matching box in this bin
                for pos_idx, (px, py, pz, pw, ph, pd) in enumerate(bins[bi]):
                    if px == bx and py == by and pz == bz and pw == bw and ph == bh and pd == bd:
                        bin_box_indices[bi].append((pos_idx, orig_idx))
                        break
        
        for b_idx in range(len(bins) - 1, 0, -1):
            if time.time() > deadline - 15.0:
                break
            if not bins[b_idx]:
                continue
            # Try to move each box in this bin to an earlier bin (largest first)
            boxes_to_try = sorted(bin_box_indices[b_idx], key=lambda x: -boxes[x[1]][0] * boxes[x[1]][1] * boxes[x[1]][2])
            for pos_idx, orig_idx in boxes_to_try:
                if time.time() > deadline - 15.0:
                    break
                if pos_idx >= len(bins[b_idx]):
                    continue
                px, py, pz, pw, ph, pd = bins[b_idx][pos_idx]
                # Remove from current bin
                bins[b_idx].pop(pos_idx)
                # Update indices in bin_box_indices
                for i in range(len(bin_box_indices[b_idx])):
                    if bin_box_indices[b_idx][i][0] == pos_idx:
                        bin_box_indices[b_idx][i] = (-1, orig_idx)
                        break
                    elif bin_box_indices[b_idx][i][0] > pos_idx:
                        bin_box_indices[b_idx][i] = (bin_box_indices[b_idx][i][0] - 1, bin_box_indices[b_idx][i][1])
                
                # Try to place in earlier bins
                moved = False
                for target_bin in range(b_idx):
                    if not bins[target_bin]:
                        continue
                    pos = find_position(bins[target_bin], pw, ph, pd)
                    if pos is not None:
                        tx, ty, tz = pos
                        new_pos_idx = len(bins[target_bin])
                        bins[target_bin].append((tx, ty, tz, pw, ph, pd))
                        bin_box_indices[target_bin].append((new_pos_idx, orig_idx))
                        placements[orig_idx] = (target_bin, tx, ty, tz, pw, ph, pd)
                        moved = True
                        break
                if not moved:
                    # Put back
                    bins[b_idx].append((px, py, pz, pw, ph, pd))
                    # Restore index
                    for i in range(len(bin_box_indices[b_idx])):
                        if bin_box_indices[b_idx][i][1] == orig_idx:
                            bin_box_indices[b_idx][i] = (len(bins[b_idx]) - 1, orig_idx)
                            break
    
    return placements
# EVOLVE-BLOCK-END
