import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    container_vol = W * H * D
    if container_vol <= 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]

    # Sort boxes by volume descending
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    placements = [None] * n
    bins = [] # List of lists of (x, y, z, w, h, d)
    grids = [] # List of 3D numpy arrays
    bin_volumes = []
    
    # Helper to get candidate positions in a bin
    def get_candidates(b_idx, w, h, d):
        # 1. Origin
        cands = [(0, 0, 0)]
        
        # 2. Surfaces of existing boxes
        for bx, by, bz, bw, bh, bd in bins[b_idx]:
            # Top
            cands.append((bx, by, bz + bd))
            # Right
            cands.append((bx + bw, by, bz))
            # Front
            cands.append((bx, by + bh, bz))
            
            # Combinations of surfaces (corners/edges)
            cands.append((bx + bw, by, bz + bd))
            cands.append((bx, by + bh, bz + bd))
            cands.append((bx + bw, by + bh, bz))
            cands.append((bx, by, bz + bd)) # Already have top, but maybe different x,y? No, same x,y.
            
            # More combinations
            cands.append((bx + bw, by + bh, bz + bd))
            cands.append((bx + bw, by + bh, bz))
            cands.append((bx + bw, by, bz + bd))
            cands.append((bx, by + bh, bz + bd))

        # Deduplicate
        unique_cands = list(set(cands))
        return unique_cands

    def fits(b_idx, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        grid = grids[b_idx]
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        return True

    def place_box(b_idx, x, y, z, w, h, d):
        bins[b_idx].append((x, y, z, w, h, d))
        grids[b_idx][x:x+w, y:y+h, z:z+d] = 1
        bin_volumes[b_idx] += w * h * d

    def new_bin():
        b_idx = len(bins)
        bins.append([])
        grids.append(np.zeros((W, H, D), dtype=np.uint8))
        bin_volumes.append(0)
        return b_idx

    # Primary Strategy: Best Fit Decreasing with "Compactness" Heuristic
    # We want to minimize B. Among existing bins, we prefer the one that results in the "most compact" shape.
    # A proxy for compactness is minimizing the "bounding box volume" of the union of placed boxes.
    # Or simply, minimizing the max extent in each dimension?
    # Let's use a cost function that penalizes extending the bounding box of the bin.
    
    def bin_bounding_box(b_idx):
        if not bins[b_idx]:
            return (0, 0, 0)
        max_x = max(b[0] + b[3] for b in bins[b_idx])
        max_y = max(b[1] + b[4] for b in bins[b_idx])
        max_z = max(b[2] + b[5] for b in bins[b_idx])
        return (max_x, max_y, max_z)

    for i, vol, w, h, d in indexed_boxes:
        if time.time() > deadline - 5:
            # Emergency: place remaining boxes in new bins or first fit
            for j, v, ww, hh, dd in indexed_boxes[i:]:
                b_idx = 0
                placed = False
                # Try first fit in existing bins
                for b in range(len(bins)):
                    if fits(b, 0, 0, 0, ww, hh, dd):
                        place_box(b, 0, 0, 0, ww, hh, dd)
                        placements[j] = (b, 0, 0, 0, ww, hh, dd)
                        placed = True
                        break
                if not placed:
                    b_idx = new_bin()
                    place_box(b_idx, 0, 0, 0, ww, hh, dd)
                    placements[j] = (b_idx, 0, 0, 0, ww, hh, dd)
            break
            
        best_bin_idx = -1
        best_pos = None
        best_cost = float('inf')
        
        # Try existing bins
        for b_idx in range(len(b_idx) if False else len(bins)):
            cands = get_candidates(b_idx, w, h, d)
            for cx, cy, cz in cands:
                if not fits(b_idx, cx, cy, cz, w, h, d):
                    continue
                
                # Calculate cost
                # 1. Check if this placement extends the bounding box
                # New bounding box
                new_max_x = max(cx + w, *( [b[0]+b[3] for b in bins[b_idx]] if bins[b_idx] else [0] ))
                new_max_y = max(cy + h, *( [b[1]+b[4] for b in bins[b_idx]] if bins[b_idx] else [0] ))
                new_max_z = max(cz + d, *( [b[2]+b[5] for b in bins[b_idx]] if bins[b_idx] else [0] ))
                
                # Current bounding box volume
                cur_bb_vol = 1 # At least 1
                if bins[b_idx]:
                    cur_max_x = max(b[0]+b[3] for b in bins[b_idx])
                    cur_max_y = max(b[1]+b[4] for b in bins[b_idx])
                    cur_max_z = max(b[2]+b[5] for b in bins[b_idx])
                    cur_bb_vol = cur_max_x * cur_max_y * cur_max_z
                    if cur_bb_vol == 0: cur_bb_vol = 1
                
                new_bb_vol = new_max_x * new_max_y * new_max_z
                if new_bb_vol == 0: new_bb_vol = 1
                
                # Cost: prefer placements that don't increase bounding box volume much
                # And prefer lower coordinates to keep it compact
                cost = new_bb_vol + 0.1 * (cx + cy + cz)
                
                # Tie-breaker: prefer bin with higher volume (more full) to consolidate
                # But cost is dominated by bb_vol.
                
                if cost < best_cost:
                    best_cost = cost
                    best_bin_idx = b_idx
                    best_pos = (cx, cy, cz)
        
        if best_bin_idx == -1:
            # No existing bin fits
            best_bin_idx = new_bin()
            best_pos = (0, 0, 0)
        
        bx, by, bz = best_pos
        place_box(best_bin_idx, bx, by, bz, w, h, d)
        placements[i] = (best_bin_idx, bx, by, bz, w, h, d)

    return [tuple(p) for p in placements]
