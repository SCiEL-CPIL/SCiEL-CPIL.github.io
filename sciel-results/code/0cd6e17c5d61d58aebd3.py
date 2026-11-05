import time
import random

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def overlaps(b1, b2):
        # b1, b2 are (x, y, z, w, h, d)
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
            
        # Also try shifting to align with other faces?
        # For now, standard "skyline" or "corner" candidates from existing boxes are usually sufficient.
        # We can add more if needed, but let's keep it efficient.
        
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
            # Calculate current max extent
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
            
            # Increase in volume of bounding box
            old_vol = max_x * max_y * max_z
            new_vol = new_max_x * new_max_y * new_max_z
            return new_vol - old_vol
        elif heuristic == 3: # Min Z, then Y, then X (Shelf-like)
            return cz * 1000000 + cy * 1000 + cx
        else:
            return cx + cy + cz

    def run_strategy(sort_key, heuristic):
        # sort_key: function to sort boxes
        # heuristic: int for scoring function
        
        indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
        indexed_boxes.sort(key=sort_key, reverse=True)
        
        bins = []
        placements = [None] * n
        
        for orig_idx, box in indexed_boxes:
            w, h, d = box
            placed = False
            
            # Try existing bins
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
                # Open new bin
                bins.append([])
                bins[-1].append((0, 0, 0, w, h, d))
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
        # Calculate score for this strategy
        if not bins:
            return float('inf'), placements
            
        # B is number of occupied bins
        B = len(bins)
        
        # Last bin is the one with the max label? 
        # "The largest label defines the last bin."
        # In our implementation, labels are 0..B-1. So last bin is bins[-1].
        # However, if we opened bins in a different order, the "largest label" might not be the last one appended.
        # But we append sequentially, so bins[-1] has the largest index.
        
        last_bin = bins[-1]
        occupied_vol = sum(b[3]*b[4]*b[5] for b in last_bin)
        container_vol = W * H * D
        fill = occupied_vol / container_vol if container_vol > 0 else 0
        
        # Score: B + 0.1 * (1 - fill)
        strategy_score = B + 0.1 * (1 - fill)
        
        return strategy_score, placements

    # Define sorting strategies
    # 1. Volume desc, MaxDim desc
    sort_vol_max = lambda x: (x[1][0]*x[1][1]*x[1][2], max(x[1]))
    # 2. MaxDim desc, Volume desc
    sort_max_vol = lambda x: (max(x[1]), x[1][0]*x[1][1]*x[1][2])
    # 3. Height (z-dim) desc, Volume desc
    sort_h_vol = lambda x: (x[1][2], x[1][0]*x[1][1]*x[1][2])
    # 4. Width (x-dim) desc, Volume desc
    sort_w_vol = lambda x: (x[1][0], x[1][0]*x[1][1]*x[1][2])
    # 5. Depth (y-dim) desc, Volume desc
    sort_d_vol = lambda x: (x[1][1], x[1][0]*x[1][1]*x[1][2])
    # 6. Surface Area desc
    sort_area = lambda x: (2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][0]*x[1][2]), max(x[1]))

    strategies = [
        (sort_vol_max, 0),
        (sort_vol_max, 1),
        (sort_vol_max, 2),
        (sort_vol_max, 3),
        (sort_max_vol, 0),
        (sort_max_vol, 1),
        (sort_max_vol, 2),
        (sort_h_vol, 0),
        (sort_h_vol, 3),
        (sort_w_vol, 0),
        (sort_d_vol, 0),
        (sort_area, 0),
    ]
    
    best_score = float('inf')
    best_placements = None
    
    for sort_key, heuristic in strategies:
        if time.time() > deadline:
            break
            
        s, placements = run_strategy(sort_key, heuristic)
        
        if s < best_score:
            best_score = s
            best_placements = placements
            
    # If no strategy finished or all failed (should not happen), fallback to simple
    if best_placements is None:
        # Fallback: simple greedy
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
                    # Pick first valid (which is 0,0,0 or best by sum)
                    # Just pick the one with min sum for fallback
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

    return best_placements
