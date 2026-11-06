"""Improved program for 3D Container Loading using Layered Shelf Packing."""
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Precompute box volumes and sort indices by volume descending
    box_vols = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(len(boxes)), key=lambda i: box_vols[i], reverse=True)
    
    results = [None] * len(boxes)
    remaining_indices = list(sorted_indices)
    
    z_current = 0
    
    # Helper to check if a 2D box (x, y, w, d) overlaps with any in placed_2d
    def check_2d_overlap(x, y, w, d, placed_2d):
        for (px, py, pw, pd) in placed_2d:
            # Check X overlap
            if not (x + w <= px or px + pw <= x):
                # Check Y overlap
                if not (y + d <= py or py + pd <= y):
                    return True
        return False

    # Helper to find a position for a 2D box (w, d) in W x D avoiding placed_2d
    def find_2d_position(w, d, placed_2d):
        x_cands = {0}
        y_cands = {0}
        
        for (px, py, pw, pd) in placed_2d:
            x_cands.add(px)
            x_cands.add(px + pw)
            y_cands.add(py)
            y_cands.add(py + pd)
            
        x_cands = sorted(x_cands)
        y_cands = sorted(y_cands)
        
        for cy in y_cands:
            if cy + d > D:
                break
            for cx in x_cands:
                if cx + w > W:
                    break
                if not check_2d_overlap(cx, cy, w, d, placed_2d):
                    return cx, cy
        return None

    while remaining_indices:
        layer_height = 0
        placed_2d = []
        new_remaining = []
        
        for idx in remaining_indices:
            w, h, d = boxes[idx]
            placed_in_layer = False
            
            # Generate unique orientations
            perms = set()
            perms.add((w, h, d))
            perms.add((w, d, h))
            perms.add((h, w, d))
            perms.add((h, d, w))
            perms.add((d, w, h))
            perms.add((d, h, w))
            
            # Filter orientations that fit in remaining height
            valid_perms = [p for p in perms if p[1] <= H - z_current and p[0] <= W and p[2] <= D]
            
            if not valid_perms:
                new_remaining.append(idx)
                continue
                
            # Sort orientations by height descending? Or volume? 
            # Usually, taller boxes first in a layer helps maximize layer height efficiency?
            # Or maybe smallest height first to fit more?
            # Let's try sorting by height descending to establish the layer height early.
            valid_perms.sort(key=lambda x: x[1], reverse=True)
            
            for (bw, bh, bd) in valid_perms:
                pos = find_2d_position(bw, bd, placed_2d)
                if pos is not None:
                    cx, cy = pos
                    placed_2d.append((cx, cy, bw, bd))
                    results[idx] = (0, cx, cy, z_current, bw, bh, bd)
                    layer_height = max(layer_height, bh)
                    placed_in_layer = True
                    break
                    
            if not placed_in_layer:
                new_remaining.append(idx)
                
        if layer_height == 0:
            # No box could be placed in this layer, break to avoid infinite loop
            break
            
        z_current += layer_height
        remaining_indices = new_remaining
        
    # Mark any remaining boxes as left out
    for idx in remaining_indices:
        w, h, d = boxes[idx]
        results[idx] = (-1, 0, 0, 0, w, h, d)
        
    return results
# EVOLVE-BLOCK-END
