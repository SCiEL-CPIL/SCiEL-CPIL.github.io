"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    deadline = time.time() + 105.0  # safety margin
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Try multiple sorting strategies and pick the best
    strategies = [
        "volume_desc",
        "max_dim_desc",
        "min_dim_desc",
        "aspect_desc",
        "volume_asc",
        "max_dim_asc",
    ]
    
    def sort_boxes(boxes, strategy):
        indexed = list(enumerate(boxes))
        if strategy == "volume_desc":
            indexed.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))
        elif strategy == "max_dim_desc":
            indexed.sort(key=lambda t: -max(t[1]))
        elif strategy == "min_dim_desc":
            indexed.sort(key=lambda t: -min(t[1]))
        elif strategy == "aspect_desc":
            indexed.sort(key=lambda t: -(max(t[1]) / min(t[1]) if min(t[1]) > 0 else 0))
        elif strategy == "volume_asc":
            indexed.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2])
        elif strategy == "max_dim_asc":
            indexed.sort(key=lambda t: max(t[1]))
        else:
            indexed.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))
        return [(i, b) for i, b in indexed]
    
    def shelf_pack_3d(boxes_sorted, W, H, D, deadline):
        """Shelf packing with rotation. Returns placements list and total volume."""
        placements = [None] * len(boxes)
        # We track placed boxes as axis-aligned boxes: (x, y, z, w, h, d)
        placed = []
        
        # Sort boxes by index for consistent output
        box_list = boxes_sorted  # list of (original_index, (w,h,d))
        
        for orig_idx, (bw, bh, bd) in box_list:
            if time.time() > deadline:
                break
            
            best_pos = None
            best_orient = None
            best_score = -1
            
            for ow, oh, od in get_orientations(bw, bh, bd):
                # Try to place at various anchor points
                # Anchor points: (0,0,0), and top of each placed box, and right of each placed box
                candidates = [(0, 0, 0)]
                for (px, py, pz, pw, ph, pd) in placed:
                    candidates.append((px + pw, py, pz))
                    candidates.append((px, py + ph, pz))
                    candidates.append((px, py, pz + pd))
                
                for (cx, cy, cz) in candidates:
                    # Check if box fits at (cx, cy, cz) with size (ow, oh, od)
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check overlap with all placed boxes
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (cx + ow <= px or px + pw <= cx or
                                cy + oh <= py or py + ph <= cy or
                                cz + od <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    # Score: prefer positions closer to origin (bottom-left-back)
                    score = -(cx * 1.0 + cy * 1.0 + cz * 1.0)
                    # Prefer orientations that align larger dims with smaller container dims?
                    # Actually, let's just use proximity to origin
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                ox, oy, oz = best_pos
                ow, oh, od = best_orient
                placed.append((ox, oy, oz, ow, oh, od))
                placements[orig_idx] = (0, ox, oy, oz, ow, oh, od)
            else:
                placements[orig_idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        # Fill any remaining None entries
        total_vol = 0
        for i in range(len(boxes)):
            if placements[i] is None:
                bw, bh, bd = boxes[i]
                placements[i] = (-1, 0, 0, 0, bw, bh, bd)
            else:
                if placements[i][0] == 0:
                    _, x, y, z, w, h, d = placements[i]
                    total_vol += w * h * d
        
        return placements, total_vol
    
    best_result = None
    best_volume = -1
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        sorted_boxes = sort_boxes(boxes, strategy)
        placements, vol = shelf_pack_3d(sorted_boxes, W, H, D, deadline)
        
        if vol > best_volume:
            best_volume = vol
            best_result = placements
    
    if best_result is None:
        # Fallback: no boxes fit
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
# EVOLVE-BLOCK-END
