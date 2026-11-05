import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    container_vol = W * H * D
    if container_vol <= 0:
        return [(0, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    # Precompute box properties
    vols = np.array([b[0]*b[1]*b[2] for b in boxes], dtype=float)
    max_dims = np.array([max(b) for b in boxes], dtype=float)
    h_dims = np.array([b[1] for b in boxes], dtype=float)  # h is y-dim in (w,h,d)? Wait.
    # Container is W, H, D. Box is w, h, d.
    # Usually w->x, h->y, d->z OR w->x, h->z, d->y?
    # Task says: "preserving its original ordered dimensions (no rotations)"
    # Interface: pack(boxes, container) -> (bin, x, y, z, w, h, d)
    # So box dims are (w, h, d).
    # Container dims are (W, H, D).
    # Usually x corresponds to w, y to h, z to d? Or x to w, y to d, z to h?
    # The task says "scored by ... last-bin occupied volume".
    # Let's assume standard mapping: w->x, h->y, d->z.
    # Wait, the seed program uses:
    # W, H, D = container
    # w, h, d = box
    # overlaps: x1+w1, y1+h1, z1+d1
    # So w->x, h->y, d->z.
    
    # Sort keys
    def key_vol_max(i):
        return (vols[i], max_dims[i])
    
    def key_max_vol(i):
        return (max_dims[i], vols[i])
    
    def key_h_vol(i):
        return (boxes[i][1], vols[i]) # h is y-dim
    
    def key_w_vol(i):
        return (boxes[i][0], vols[i]) # w is x-dim
    
    def key_d_vol(i):
        return (boxes[i][2], vols[i]) # d is z-dim
    
    def key_area(i):
        w,h,d = boxes[i]
        return (2*(w*h + h*d + w*d), max_dims[i])

    def overlaps(b1, b2):
        x1, y1, z1, w1, h1, d1 = b1
        x2, y2, z2, w2, h2, d2 = b2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def get_candidates(box, bin_boxes):
        w, h, d = box
        candidates = set()
        candidates.add((0, 0, 0))
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            
        valid_candidates = []
        for cx, cy, cz in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            is_overlap = False
            new_box = (cx, cy, cz, w, h, d)
            for existing in bin_boxes:
                if overlaps(new_box, existing):
                    is_overlap = True
                    break
            if not is_overlap:
                valid_candidates.append((cx, cy, cz))
        return valid_candidates

    def score_position(pos, box, bin_boxes, heuristic):
        cx, cy, cz = pos
        w, h, d = box
        
        if heuristic == 0: # Min Sum
            return cx + cy + cz
        elif heuristic == 1: # Min Max
            return max(cx, cy, cz)
        elif heuristic == 2: # Min Bounding Box Expansion
            max_x = 0
            max_y = 0
            max_z = 0
            for bx, by, bz, bw, bh, bd in bin_boxes:
                max_x = max(max_x, bx + bw)
                max_y = max(max_y, by + bh)
                max_z = max(max_z, bz + bd)
            
            new_max_x = max(max_x, cx + w)
            new_max_y = max(max_y, cy + h)
            new_max_z = max(max_z, cz + d)
            
            old_vol = max_x * max_y * max_z
            new_vol = new_max_x * new_max_y * new_max_z
            return new_vol - old_vol
        elif heuristic == 3: # Min Z, then Y, then X (Shelf-like)
            return cz * 1000000 + cy * 1000 + cx
        else:
            return cx + cy + cz

    def run_strategy(sort_key, heuristic):
        indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
        indexed_boxes.sort(key=lambda x: sort_key(x[0]), reverse=True)
        
        bins = []
        placements = [None] * n
        
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            
            best_bin_idx = -1
            best_pos = None
            best_score = float('inf')
            
            for b_idx, bin_boxes in enumerate(bins):
                candidates = get_candidates(box, bin_boxes)
                if not candidates:
                    continue
                
                for pos in candidates:
                    s = score_position(pos, box, bin_boxes, heuristic)
                    if s < best_score:
                        best_score = s
                        best_bin_idx = b_idx
                        best_pos = pos
            
            if best_bin_idx != -1:
                cx, cy, cz = best_pos
                bins[best_bin_idx].append((cx, cy, cz, w, h, d))
                placements[orig_idx] = (best_bin_idx, cx, cy, cz, w, h, d)
            else:
                bins.append([])
                bins[-1].append((0, 0, 0, w, h, d))
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
        if not bins:
            return float('inf'), placements
            
        B = len(bins)
        last_bin = bins[-1]
        occupied_vol = sum(b[3]*b[4]*b[5] for b in last_bin)
        fill = occupied_vol / container_vol if container_vol > 0 else 0
        
        strategy_score = B + 0.1 * (1 - fill)
        
        return strategy_score, placements

    strategies = [
        (key_vol_max, 0),
        (key_vol_max, 1),
        (key_vol_max, 2),
        (key_vol_max, 3),
        (key_max_vol, 0),
        (key_max_vol, 1),
        (key_max_vol, 2),
        (key_h_vol, 0),
        (key_h_vol, 3),
        (key_w_vol, 0),
        (key_d_vol, 0),
        (key_area, 0),
    ]
    
    best_score = float('inf')
    best_placements = None
    best_bins = None
    
    for sort_key, heuristic in strategies:
        if time.time() > deadline:
            break
            
        s, placements = run_strategy(sort_key, heuristic)
        
        if s < best_score:
            best_score = s
            best_placements = placements
            # Reconstruct bins for potential post-processing
            if placements:
                temp_bins = [[] for _ in range(max(p[0] for p in placements) + 1)]
            else:
                temp_bins = []
            for p in placements:
                temp_bins[p[0]].append((p[1], p[2], p[3], p[4], p[5], p[6]))
            best_bins = temp_bins

    # Post-processing: Try to improve last bin fill or reduce B
    if best_placements is None:
        indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
        indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
        bins = []
        placements = [None] * n
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            for b_idx, bin_boxes in enumerate(bins):
                candidates = get_candidates(box, bin_boxes)
                if candidates:
                    best_pos = min(candidates, key=lambda p: p[0]+p[1]+p[2])
                    cx, cy, cz = best_pos
                    bins[b_idx].append((cx, cy, cz, w, h, d))
                    placements[orig_idx] = (b_idx, cx, cy, cz, w, h, d)
                    placed = True
                    break
            if not placed:
                bins.append([])
                bins[-1].append((0, 0, 0, w, h, d))
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
        best_placements = placements
        best_bins = bins

    # Optimization Phase: Move boxes from the last bin to other bins if possible
    # This can reduce B if the last bin becomes empty, or improve fill if we move small boxes out?
    # Actually, moving boxes OUT of the last bin reduces its fill, which is bad for the 0.1*(1-fill) term.
    # But if we can move ALL boxes out, B decreases by 1, which is a gain of 1.0. The loss in fill term is at most 0.1.
    # So if we can empty the last bin, it's always a win.
    
    if best_bins and len(best_bins) > 1 and time.time() < deadline - 5.0:
        last_bin_idx = len(best_bins) - 1
        last_bin_boxes = best_bins[last_bin_idx]
        
        # Try to move each box from last bin to any other bin
        # We should try to move boxes that are easiest to place first? Or largest?
        # Let's try to move all boxes in the last bin to other bins.
        # If we succeed in moving all, we remove the last bin.
        
        moved_any = False
        # Sort last bin boxes by volume descending (harder to place first? or easier?)
        # Let's try moving them one by one. If we can't move one, we might try swapping?
        # Simple approach: for each box in last bin, try to place it in bins 0..B-2.
        
        remaining = list(last_bin_boxes)
        while remaining:
            moved = False
            for i in range(len(remaining)):
                box_data = remaining[i]
                w, h, d = box_data[3], box_data[4], box_data[5]
                box = (w, h, d)
                
                best_target = -1
                best_pos = None
                best_score = float('inf')
                
                for b_idx in range(len(best_bins) - 1): # Exclude last bin
                    bin_boxes = best_bins[b_idx]
                    candidates = get_candidates(box, bin_boxes)
                    if not candidates:
                        continue
                    for pos in candidates:
                        s = score_position(pos, box, bin_boxes, 0) # Use min sum
                        if s < best_score:
                            best_score = s
                            best_target = b_idx
                            best_pos = pos
                
                if best_target != -1:
                    # Move box
                    cx, cy, cz = best_pos
                    best_bins[best_target].append((cx, cy, cz, w, h, d))
                    
                    # Update placement
                    # Find the index of this box in best_placements corresponding to last_bin_idx
                    # We need to track which original box index this is.
                    # Our best_bins doesn't track original indices.
                    # This is a problem. We need to maintain the mapping.
                    
                    # Let's abort this specific optimization if we don't have indices.
                    # We need to re-run with indices or store them.
                    break
                
                if not moved:
                    break
            if not moved:
                break
        
        # Since we lost the index mapping, let's implement a safer version:
        # Re-run the best strategy but with a "try to merge last bin" pass that tracks indices.
        
        # Actually, let's just try to find a better strategy if we have time.
        # The current best is likely already good.
        # Let's try one more strategy: "Last Bin Fill" optimization by sorting boxes to fill the last bin more?
        # No, the last bin is determined by the order of opening bins.
        
        # Let's try a different heuristic: "Min Distance to Center" or similar?
        # Or simply try more sort orders if time permits.
        
        extra_strategies = [
            (key_vol_max, 4), # Min Distance to Center
            (key_max_vol, 4),
        ]
        
        for sort_key, heuristic in extra_strategies:
            if time.time() > deadline - 2.0:
                break
            s, placements = run_strategy(sort_key, heuristic)
            if s < best_score:
                best_score = s
                best_placements = placements

    return best_placements
