import time
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Preprocess boxes: Sort by volume descending (FFD)
    # We need to keep track of original indices to return results in original order
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Store placements in original index order
    # placement[i] = (bin, x, y, z, w, h, d)
    # bin=0 if placed, -1 if not
    placements = [None] * n
    
    # 2. Initialize Extreme Points
    # Each point is (x, y, z)
    extreme_points = [(0, 0, 0)]
    
    # List of placed boxes: (x, y, z, w, h, d)
    placed_boxes = []
    
    # 3. Iterate through sorted boxes
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, mark remaining as not placed
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
            
        # Get unique orientations for this box
        # Dimensions are (w, h, d). Permutations:
        # (w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)
        # Remove duplicates
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        
        # Try each extreme point
        # To make it deterministic and potentially better, we can sort extreme points
        # But usually just iterating is fine. 
        # We want to find *a* valid placement. 
        # Heuristic: Prefer positions that are "lower" (smaller x+y+z) to fill bottom layers first?
        # Or prefer positions that are "tighter"?
        # Let's try to find the placement that minimizes the bounding box of all placed items?
        # That's expensive. 
        # Simple heuristic: First valid placement found in a specific order.
        # Let's sort extreme points by (x, y, z) to fill from origin.
        
        # Optimization: If extreme points list grows large, it can be slow.
        # With 60 boxes, max extreme points is roughly 60*3 = 180.
        # 180 points * 6 orientations * 60 placed boxes (for overlap check) = 64,800 operations per box.
        # 60 boxes * 64,800 = ~4M operations. This is very fast in Python (<1s).
        
        for px, py, pz in extreme_points:
            if time.time() > deadline:
                break
                
            for ow, oh, od in orients:
                # Check bounds
                if px + ow > W or py + oh > H or pz + od > D:
                    continue
                
                # Check overlap with all placed boxes
                valid = True
                for bx, by, bz, bw, bh, bd in placed_boxes:
                    # Check if (px, py, pz, ow, oh, od) overlaps (bx, by, bz, bw, bh, bd)
                    if (px < bx + bw and bx < px + ow and
                        py < by + bh and by < py + oh and
                        pz < bz + bd and bz < pz + od):
                        valid = False
                        break
                
                if valid:
                    # Found a valid placement
                    # Score it. We want to maximize density.
                    # A common heuristic is to minimize the "wasted space" or keep the skyline low.
                    # Let's use a simple score: x + y + z. Lower is better (fills from bottom-front).
                    # Or maybe we should prioritize placing large boxes in corners?
                    # Let's stick to First Fit with sorted points.
                    # To make it "Best Fit" like, we could evaluate all and pick the best.
                    # Let's pick the one with minimum (x+y+z). If tie, max volume?
                    score = px + py + pz
                    if score < best_score:
                        best_score = score
                        best_placement = (px, py, pz, ow, oh, od)
                        
                    # Optional: Break early if we find a "perfect" fit at (0,0,0)?
                    # No, other orientations might be better for future.
                    # But for speed, we can break if we find a valid one at the first sorted point?
                    # Let's collect all valid ones and pick the best score to improve packing.
        
        if best_placement is not None:
            px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            placed_boxes.append((px, py, pz, ow, oh, od))
            
            # Generate new extreme points
            # The new box occupies [px, px+ow] x [py, py+oh] x [pz, pz+od]
            # New potential positions for next boxes are the corners of this box that are "exposed"
            # (px+ow, py, pz)
            # (px, py+oh, pz)
            # (px, py, pz+od)
            # Also combinations like (px+ow, py+oh, pz)?
            # Standard EP algorithm adds (x+w, y, z), (x, y+h, z), (x, y, z+d)
            new_points = [
                (px + ow, py, pz),
                (px, py + oh, pz),
                (px, py, pz + od)
            ]
            
            # Add new points to extreme_points, avoiding duplicates
            for p in new_points:
                if p not in extreme_points:
                    extreme_points.append(p)
        else:
            # Box could not be placed
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
