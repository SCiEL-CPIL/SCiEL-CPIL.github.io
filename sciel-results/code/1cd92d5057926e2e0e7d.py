"""Model-and-optimize 3D bin packing with deadline guard."""
import time
import random
import math
from typing import List, Tuple

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    
    # Sort indices by volume descending
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)
    
    def find_position(existing_items, w, h, d, W, H, D):
        """
        Find a position (x, y, z) where box (w, h, d) fits without overlap.
        Uses all unique coordinate combinations from box faces.
        """
        if not existing_items:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Collect unique x, y, z coordinates from box faces
        x_coords = {0, W}
        y_coords = {0, H}
        z_coords = {0, D}
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            x_coords.add(ex)
            x_coords.add(ex + ew)
            y_coords.add(ey)
            y_coords.add(ey + eh)
            z_coords.add(ez)
            z_coords.add(ez + ed)
        
        # Filter to only coordinates where the box would fit
        x_coords = sorted(x for x in x_coords if x + w <= W)
        y_coords = sorted(y for y in y_coords if y + h <= H)
        z_coords = sorted(z for z in z_coords if z + d <= D)
        
        if not x_coords or not y_coords or not z_coords:
            return None
        
        # Generate candidate positions and sort by (z, y, x) - bottom-left-front
        candidates = []
        for cx in x_coords:
            for cy in y_coords:
                for cz in z_coords:
                    candidates.append((cx, cy, cz))
        
        candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        
        for (cx, cy, cz) in candidates:
            # Check overlap with all existing items
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in existing_items:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                return (cx, cy, cz)
        
        return None
    
    def bin_fill_fraction(bin_items, W, H, D):
        """Compute the volume fraction of a bin that is filled."""
        vol = sum(ew * eh * ed for (_, _, _, ew, eh, ed) in bin_items)
        return vol / (W * H * D)
    
    def try_place(placement_state, box_idx):
        """
        Try to place box_idx in existing bins first (best fit), then new bin.
        Returns (success, new_state, bin_id, pos)
        """
        w, h, d = boxes[box_idx]
        
        # Try existing bins, prefer the one that results in highest fill fraction
        best_bin = None
        best_pos = None
        best_fill = -1.0
        
        for bin_id in sorted(placement_state.keys()):
            if time.time() > deadline:
                break
            pos = find_position(placement_state[bin_id], w, h, d, W, H, D)
            if pos is not None:
                # Compute fill fraction after placing this box
                current_vol = sum(ew * eh * ed for (_, _, _, ew, eh, ed) in placement_state[bin_id])
                new_fill = (current_vol + w * h * d) / (W * H * D)
                if new_fill > best_fill:
                    best_fill = new_fill
                    best_bin = bin_id
                    best_pos = pos
        
        if best_bin is not None:
            new_bin_list = placement_state[best_bin] + [(best_pos[0], best_pos[1], best_pos[2], w, h, d)]
            new_state = dict(placement_state)
            new_state[best_bin] = new_bin_list
            return True, new_state, best_bin, best_pos
        
        # Try new bin
        new_bin_id = max(placement_state.keys()) + 1 if placement_state else 0
        new_state = dict(placement_state)
        new_state[new_bin_id] = [(0, 0, 0, w, h, d)]
        return True, new_state, new_bin_id, (0, 0, 0)
    
    # Main optimization: try multiple orderings
    best_result = None
    best_score = float('inf')
    
    orderings = [
        indices_sorted,  # volume descending
        list(reversed(indices_sorted)),  # volume ascending
        indices,  # original order
        sorted(indices, key=lambda i: max(boxes[i]), reverse=True),  # max dim descending
        sorted(indices, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True),  # volume desc (same as first)
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        state = {}
        placements = [None] * n
        
        for box_idx in ordering:
            if time.time() > deadline:
                break
            success, state, bin_id, pos = try_place(state, box_idx)
            if success:
                placements[box_idx] = (bin_id, pos[0], pos[1], pos[2], boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2])
        
        # Check if all placed
        if all(p is not None for p in placements):
            score = compute_score(placements, W, H, D)
            if score < best_score:
                best_score = score
                best_result = placements
    
    # If no valid result found, fall back to simple shelf
    if best_result is None or any(p is None for p in best_result):
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements
    
    # Post-process: relabel bins to be consecutive starting from 0
    if best_result:
        bin_labels = sorted(set(p[0] for p in best_result))
        label_map = {old: new for new, old in enumerate(bin_labels)}
        best_result = [(label_map[p[0]], p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result]
    
    return best_result


def compute_score(placements, W, H, D):
    if not placements:
        return 0.0
    
    bins_used = set(p[0] for p in placements)
    B = len(bins_used)
    
    last_bin = max(bins_used)
    
    container_vol = W * H * D
    last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
    
    score = B + 0.1 * (1 - last_bin_vol / container_vol)
    return score
