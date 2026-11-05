import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Precompute volumes and sort indices
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    
    # We will try multiple strategies and keep the best result
    # Strategy components:
    # 1. Sorting order
    # 2. Placement heuristic (which position to pick among candidates)
    # 3. Bin selection heuristic (which bin to pick if multiple fit)

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
        elif heuristic == 4: # Min Z, then Min Sum (Hybrid)
            return cz * 10000 + cx + cy
        else:
            return cx + cy + cz

    def run_strategy(sort_key, pos_heuristic, bin_heuristic):
        """
        sort_key: function to sort boxes
        pos_heuristic: int for scoring position within a bin
        bin_heuristic: int for scoring which bin to choose
                       0: first fit
                       1: best fit (minimize wasted space in that bin? or just min pos score)
                       2: min bounding box volume increase of the bin
        """
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
                
                best_pos_in_bin = None
                best_pos_score_in_bin = float('inf')
                
                for pos in candidates:
                    s = score_position(pos, box, bin_boxes, pos_heuristic)
                    if s < best_pos_score_in_bin:
                        best_pos_score_in_bin = s
                        best_pos_in_bin = pos
                
                if best_pos_in_bin is not None:
                    # Now decide if this bin is better than the current best_bin
                    if bin_heuristic == 0:
                        # First fit: just take the first valid bin
                        # But we already found a valid pos. 
                        # To make it deterministic "first fit", we could break here, 
                        # but let's compare scores to be consistent with other heuristics
                        # Actually, first fit usually means "don't look at other bins".
                        # Let's implement a soft first-fit: prefer lower bin index if scores are close?
                        # No, standard First Fit Decreasing: place in first bin that fits.
                        # However, the "best position" within that bin matters.
                        # Let's stick to: find best pos in this bin.
                        # Then compare with best_bin.
                        # For "First Fit" semantics, we should probably break out of bin loop if we find *any* valid bin?
                        # But that might lead to suboptimal packing if an earlier bin has a bad spot and a later bin has a great spot.
                        # The current code structure compares all bins. Let's keep that but change the score.
                        
                        # For bin_heuristic 0, let's just use the position score as the global score, 
                        # but add a small penalty for higher bin index to encourage filling earlier bins?
                        # Or simply: if bin_heuristic is 0, we break on the first valid bin found?
                        # Let's try: if bin_heuristic == 0, we break the bin loop immediately after finding a valid position in the first valid bin.
                        # But we need to find the BEST position in that bin first.
                        
                        # Let's refine: 
                        # If bin_heuristic == 0: Break out of bin loop after processing the FIRST bin that has any valid candidate.
                        # This is true First Fit.
                        
                        # Wait, the loop iterates b_idx 0..N.
                        # If we want First Fit, we should break as soon as we find a bin that fits.
                        # So:
                        if best_bin_idx == -1:
                            best_bin_idx = b_idx
                            best_pos = best_pos_in_bin
                            # Don't update best_score, just break?
                            # Yes, for First Fit.
                            break
                            
                    elif bin_heuristic == 1:
                        # Best Fit: minimize the "tightness" or "waste".
                        # Here we use the position score as the primary metric for bin comparison.
                        # This effectively chooses the bin where the box fits "lowest" or "tightest" according to pos_heuristic.
                        if best_pos_score_in_bin < best_score:
                            best_score = best_pos_score_in_bin
                            best_bin_idx = b_idx
                            best_pos = best_pos_in_bin
                            
                    elif bin_heuristic == 2:
                        # Minimize Bounding Box Volume Increase of the Bin
                        # Calculate current bin BB volume
                        max_x = 0; max_y = 0; max_z = 0
                        for bx, by, bz, bw, bh, bd in bin_boxes:
                            max_x = max(max_x, bx + bw)
                            max_y = max(max_y, by + bh)
                            max_z = max(max_z, bz + bd)
                        
                        # Find the min BB increase for this box in this bin
                        min_increase = float('inf')
                        for pos in candidates:
                            cx, cy, cz = pos
                            new_max_x = max(max_x, cx + w)
                            new_max_y = max(max_y, cy + h)
                            new_max_z = max(max_z, cz + d)
                            old_vol = max_x * max_y * max_z
                            new_vol = new_max_x * new_max_y * new_max_z
                            increase = new_vol - old_vol
                            if increase < min_increase:
                                min_increase = increase
                                # Also track the pos that gives this min increase?
                                # Or just use min_increase for bin selection, and pos_heuristic for final pos?
                                # Let's use the pos that minimizes BB increase as the best_pos_in_bin for this heuristic.
                                best_pos_in_bin = pos
                                best_pos_score_in_bin = increase # Reusing score var for consistency in comparison below?
                                # Actually, let's just compare min_increase directly.
                        
                        if min_increase < best_score:
                            best_score = min_increase
                            best_bin_idx = b_idx
                            best_pos = best_pos_in_bin

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
            
        B = len(bins)
        last_bin = bins[-1]
        occupied_vol = sum(b[3]*b[4]*b[5] for b in last_bin)
        container_vol = W * H * D
        fill = occupied_vol / container_vol if container_vol > 0 else 0
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
    # 7. Volume desc only
    sort_vol = lambda x: (x[1][0]*x[1][1]*x[1][2],)

    # Define strategies: (sort_key, pos_heuristic, bin_heuristic)
    # We will test a combination of these.
    strategies = [
        (sort_vol_max, 0, 0),  # Vol/Max, MinSum, FirstFit
        (sort_vol_max, 0, 1),  # Vol/Max, MinSum, BestFit
        (sort_vol_max, 2, 1),  # Vol/Max, MinBBExp, BestFit
        (sort_vol_max, 3, 0),  # Vol/Max, Shelf, FirstFit
        (sort_vol_max, 4, 1),  # Vol/Max, Hybrid, BestFit
        
        (sort_max_vol, 0, 0),  # Max/Vol, MinSum, FirstFit
        (sort_max_vol, 0, 1),  # Max/Vol, MinSum, BestFit
        (sort_max_vol, 2, 1),  # Max/Vol, MinBBExp, BestFit
        
        (sort_h_vol, 3, 0),    # H/Vol, Shelf, FirstFit
        (sort_h_vol, 0, 1),    # H/Vol, MinSum, BestFit
        
        (sort_w_vol, 0, 1),    # W/Vol, MinSum, BestFit
        (sort_d_vol, 0, 1),    # D/Vol, MinSum, BestFit
        (sort_area, 0, 1),     # Area, MinSum, BestFit
        (sort_vol, 0, 0),      # Vol, MinSum, FirstFit
        (sort_vol, 2, 2),      # Vol, MinBBExp, MinBBBin
    ]
    
    best_score = float('inf')
    best_placements = None
    
    for sort_key, pos_h, bin_h in strategies:
        if time.time() > deadline:
            break
            
        s, placements = run_strategy(sort_key, pos_h, bin_h)
        
        if s < best_score:
            best_score = s
            best_placements = placements
            
    # If no strategy finished or all failed, fallback to simple
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
