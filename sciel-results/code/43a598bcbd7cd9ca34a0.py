"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # 10s safety margin before 120s kill
    
    # Precompute all unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Try multiple sorting strategies and return the best packing
    def shelf_pack(sorted_boxes, orientations_list):
        """
        3D shelf packing:
        - Process boxes in order
        - Each box is placed at the lowest possible z, then lowest y, then lowest x
        - Within a "layer" (z-slice), boxes are placed on shelves (rows in y)
        - Uses a simple bottom-left-front heuristic with shelf constraints
        """
        placements = []
        # Track occupied space using a list of placed boxes
        # For efficiency, we track shelves per layer
        
        # We'll use a simple approach: maintain a list of (x, y, z, w, h, d) for placed boxes
        # and for each new box, find the best position
        
        placed = []  # list of (x, y, z, w, h, d)
        
        for i, (bw, bh, bd) in enumerate(sorted_boxes):
            if time.time() > deadline:
                break
            
            orients = orientations_list[i]
            best_pos = None
            best_score = None
            
            for ow, oh, od in orients:
                # Check if this orientation fits in container
                if ow > W or oh > H or od > D:
                    continue
                
                # Try to find a valid position
                # Strategy: try z from 0 upward, then y from 0, then x from 0
                # For each z, we consider shelves defined by existing boxes
                
                # Collect candidate z positions: 0 and top of any box that doesn't overlap in x,y
                # Simplified: try z = 0, and z = top of each placed box
                z_candidates = set([0])
                for px, py, pz, pw, ph, pd in placed:
                    # If the new box (at some x,y) doesn't overlap with this box in x,y,
                    # we can place it on top of this box
                    # But we don't know x,y yet, so we'll just consider all tops
                    z_candidates.add(pz + pd)
                
                for z in sorted(z_candidates):
                    if z + od > D:
                        continue
                    
                    # At this z, find available x,y positions
                    # Try y from 0 upward
                    y_candidates = set([0])
                    for px, py, pz, pw, ph, pd in placed:
                        if pz + pd > z or pz < z + od:
                            # This box is at the same z-level, so it defines a y boundary
                            y_candidates.add(py + ph)
                    
                    for y in sorted(y_candidates):
                        if y + oh > H:
                            continue
                        
                        # Try x from 0 upward
                        x_candidates = set([0])
                        for px, py, pz, pw, ph, pd in placed:
                            if py + ph > y or py < y + oh:
                                if pz + pd > z or pz < z + od:
                                    x_candidates.add(px + pw)
                        
                        for x in sorted(x_candidates):
                            if x + ow > W:
                                continue
                            
                            # Check for overlap with all placed boxes
                            valid = True
                            for px, py, pz, pw, ph, pd in placed:
                                # Check overlap in all 3 dimensions
                                if (x < px + pw and x + ow > px and
                                    y < py + ph and y + oh > py and
                                    z < pz + pd and z + od > pz):
                                    valid = False
                                    break
                            
                            if valid:
                                # Score: prefer lower z, then lower y, then lower x
                                score = (z, y, x)
                                if best_score is None or score < best_score:
                                    best_score = score
                                    best_pos = (x, y, z, ow, oh, od)
            
            if best_pos is not None:
                x, y, z, ow, oh, od = best_pos
                placed.append((x, y, z, ow, oh, od))
                placements.append((0, x, y, z, ow, oh, od))
            else:
                placements.append((-1, 0, 0, 0, bw, bh, bd))
        
        return placements
    
    def compute_score(placements):
        total_vol = sum(w * h * d for b, x, y, z, w, h, d in placements if b == 0)
        return total_vol / (W * H * D)
    
    # Generate different sorting strategies
    strategies = []
    
    # Strategy 1: Sort by volume descending
    def sort_by_volume_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Strategy 2: Sort by max dimension descending
    def sort_by_max_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    # Strategy 3: Sort by min dimension descending
    def sort_by_min_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)
    
    # Strategy 4: Sort by surface area descending
    def sort_by_surface_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    # Strategy 5: Sort by aspect ratio (max/min) ascending, then volume descending
    def sort_by_aspect_vol(boxes):
        def key(i):
            w, h, d = boxes[i]
            aspect = max(w, h, d) / min(w, h, d)
            vol = w * h * d
            return (aspect, -vol)
        return sorted(range(len(boxes)), key=key)
    
    strategies = [
        ("vol_desc", sort_by_volume_desc),
        ("max_dim_desc", sort_by_max_dim_desc),
        ("min_dim_desc", sort_by_min_dim_desc),
        ("surface_desc", sort_by_surface_desc),
        ("aspect_vol", sort_by_aspect_vol),
    ]
    
    # Precompute orientations for all boxes
    orientations_list = []
    for w, h, d in boxes:
        orientations_list.append(get_orientations(w, h, d))
    
    best_placements = None
    best_score = 0
    
    for strat_name, sort_func in strategies:
        if time.time() > deadline:
            break
        
        order = sort_func(boxes)
        sorted_boxes = [boxes[i] for i in order]
        sorted_orients = [orientations_list[i] for i in order]
        
        placements = shelf_pack(sorted_boxes, sorted_orients)
        
        # Reorder placements back to original box order
        # We need to map: sorted index -> original index
        # placements[k] corresponds to sorted_boxes[k] = boxes[order[k]]
        final_placements = [None] * len(boxes)
        for k, orig_idx in enumerate(order):
            final_placements[orig_idx] = placements[k]
        
        # Any boxes not placed (if we broke early) get -1
        for i in range(len(boxes)):
            if final_placements[i] is None:
                w, h, d = boxes[i]
                final_placements[i] = (-1, 0, 0, 0, w, h, d)
        
        score = compute_score(final_placements)
        if score > best_score:
            best_score = score
            best_placements = final_placements
    
    # If no strategy completed, use the base naive approach as fallback
    if best_placements is None:
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements

# EVOLVE-BLOCK-END
