import time
import random
from itertools import product

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State: list of bins. Each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_idx, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_position(bin_idx, w, h, d):
        # Generate candidate positions
        # 1. (0,0,0)
        # 2. For each box, try the 7 corners "above" or "next to" it
        #    (ox+ow, oy, oz), (ox, oy+oh, oz), (ox, oy, oz+od), etc.
        
        candidates = [(0, 0, 0)]
        
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            # Right, Front, Top
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            
            # Combinations
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
            
        # Deduplicate
        unique_candidates = list(set(candidates))
        
        # Sort candidates to prefer "tighter" packing.
        # Heuristic: Minimize z, then y, then x. This pushes boxes to the bottom-front-left.
        unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in unique_candidates:
            if can_place(bin_idx, cx, cy, cz, w, h, d):
                return cx, cy, cz
                
        return None

    def place_box(w, h, d):
        # 1. Try to place in existing bins (Best Fit)
        best_bin_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i in range(len(bins)):
            pos = find_position(i, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: minimize z, then y, then x
                score = (z, y, x)
                if score < best_score:
                    best_score = score
                    best_bin_idx = i
                    best_pos = pos
        
        if best_bin_idx != -1:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            return best_bin_idx
        
        # 2. If no existing bin works, open a new bin
        new_bin_idx = len(bins)
        bins.append([])
        bins[new_bin_idx].append((0, 0, 0, w, h, d))
        return new_bin_idx

    # Process boxes
    placements = [None] * n
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
        
        b = place_box(w, h, d)
        # Get the position from the last item in the bin
        pos = bins[b][-1]
        placements[idx] = (b, pos[0], pos[1], pos[2], w, h, d)
        
    # Handle any unplaced boxes if we broke out early
    unplaced_indices = [i for i in range(n) if placements[i] is None]
    for i in unplaced_indices:
        w, h, d = boxes[i]
        b = len(bins)
        bins.append([(0, 0, 0, w, h, d)])
        placements[i] = (b, 0, 0, 0, w, h, d)
        
    return placements
