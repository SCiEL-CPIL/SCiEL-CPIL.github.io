import time

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
    bin_volumes = []
    
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
        # 3. Also try aligning with the "sky" (max z) for each x,y combination? 
        #    A simple heuristic is to try the "lowest" possible z for a given x,y.
        #    But we don't have a grid. 
        #    Let's stick to the corner expansion which is standard for this type of heuristic.
        
        candidates = [(0, 0, 0)]
        
        # Collect unique x, y coordinates from existing boxes to build a grid of candidates?
        # Or just expand from each box.
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
            
            # Also try "on top" of the max z at this x,y if we had a height map.
            # Without a height map, we rely on the explicit corner expansion.
            # However, a common improvement is to also check (ox, oy, max_z_so_far)
            # But max_z_so_far depends on x,y.
            
        # Deduplicate
        unique_candidates = list(set(candidates))
        
        # Sort candidates to prefer "tighter" packing.
        # Heuristic: Minimize x, then y, then z. This pushes boxes to the origin.
        # Alternative: Minimize (x+y+z) or volume of the bounding box defined by this placement.
        # Let's try sorting by (x, y, z) which is a standard "First Fit" behavior in 3D.
        # To make it "Best Fit", we might want to evaluate all candidates and pick the one
        # that minimizes the "waste". 
        # Since we want to return the *best* position for the Best-Fit heuristic,
        # we should evaluate all valid candidates and return the one with the "best" score.
        # What is the best score?
        # Lower z is generally better (saves vertical space for future boxes).
        # Then lower y, then lower x.
        # Or, minimize the sum of distances to the walls?
        # Let's use (z, y, x) as the sort key for candidates in a specific bin, 
        # but since we are doing Best-Fit across bins, we need a score for the position.
        
        # For now, let's just return the first valid candidate in a sorted order.
        # To improve, we can sort by a "waste" metric.
        # Let's sort by (x, y, z) to keep it simple and fast, then the Best-Fit logic
        # across bins will pick the bin where this "tightest" fit is best.
        # Actually, if we sort by (x,y,z), we might pick a position that is tight in x but loose in z.
        # A better score for a position might be the volume of the space "wasted" below it?
        # Or simply, prefer lower Z.
        
        unique_candidates.sort(key=lambda p: (p[2], p[1], p[0])) # Min Z, then Y, then X
        
        for cx, cy, cz in unique_candidates:
            if can_place(bin_idx, cx, cy, cz, w, h, d):
                return cx, cy, cz
                
        return None

    def evaluate_bin_fit(bin_idx, w, h, d):
        """
        Returns (fits, score) where lower score is better (Best Fit).
        """
        pos = find_position(bin_idx, w, h, d)
        if pos is None:
            return False, float('inf')
        x, y, z = pos
        # Heuristic: prefer positions closer to the origin (0,0,0)
        # This tends to fill the bin from the bottom-left corner up.
        # A more sophisticated score could be the volume of the bounding box
        # created by adding this box to the bin, compared to the current bounding box.
        # But x+y+z is a good proxy for "tightness".
        score = x + y + z
        return True, score

    def place_box(w, h, d):
        # 1. Try to place in existing bins (Best Fit)
        best_bin_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i in range(len(bins)):
            fits, score = evaluate_bin_fit(i, w, h, d)
            if fits and score < best_score:
                best_score = score
                best_bin_idx = i
                best_pos = find_position(i, w, h, d) # Recalculate to be safe, or store it
                
        if best_bin_idx != -1:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            bin_volumes[best_bin_idx] += w * h * d
            return best_bin_idx
        
        # 2. If no existing bin works, open a new bin
        new_bin_idx = len(bins)
        bins.append([])
        bin_volumes.append(w * h * d)
        # Place at (0,0,0)
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
    if time.time() > deadline:
        unplaced_indices = [i for i in range(n) if placements[i] is None]
        # For each unplaced, assign to a new bin at (0,0,0)
        # This guarantees validity but poor score.
        for i in unplaced_indices:
            w, h, d = boxes[i]
            b = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placements[i] = (b, 0, 0, 0, w, h, d)
            
    return placements
