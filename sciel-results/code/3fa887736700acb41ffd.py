import time
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    W, H, D = container
    n = len(boxes)
    
    # Preprocess: Sort boxes by a heuristic weight
    # We prioritize larger volumes, but also consider "bulkiness"
    # A common heuristic is volume, but sometimes sorting by max dimension or surface area helps.
    # Let's stick to volume descending, which is standard and effective.
    indices = list(range(n))
    indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # bins: list of lists of (box_index, x, y, z, w, h, d)
    bins = []
    
    def get_candidates(occupied):
        """
        Generate candidate bottom-left-back positions for a new box in a bin.
        We generate coordinates based on the "corners" of existing boxes.
        """
        if not occupied:
            return [(0, 0, 0)]
        
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Collect all unique x, y, z coordinates from existing boxes
        # and their offsets (x+w, y+h, z+d)
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (_, x, y, z, w, h, d) in occupied:
            xs.add(x)
            xs.add(x + w)
            ys.add(y)
            ys.add(y + h)
            zs.add(z)
            zs.add(z + d)
            
        # To keep the number of candidates manageable, we don't use the full cartesian product
        # if the set is large, but for 40 boxes, it's usually fine.
        # However, a smarter approach is to only consider candidates that are "supported".
        # For simplicity and speed, we'll use the cartesian product of a limited set.
        # If the sets are too large, we might miss some tight spots, but it's fast.
        
        # A better heuristic: For each existing box, add its "top-right-front" corner
        # and combinations with 0.
        # The previous method did this. Let's stick to a refined version.
        
        # Let's re-evaluate. The full cartesian product of unique coords can be O(N^3).
        # For N=40, 40^3 = 64000 candidates. This is acceptable per placement if fast.
        # But we can be smarter.
        
        # Let's use the method from the seed but ensure we include all relevant corners.
        # The seed method added specific corners per box.
        # Let's use the cartesian product of unique coordinates, but limit it if it gets too big.
        
        if len(xs) * len(ys) * len(zs) < 5000:
            for x in xs:
                for y in ys:
                    for z in zs:
                        candidates.add((x, y, z))
        else:
            # Fallback to the per-box corner generation to limit candidate count
            for (_, x, y, z, w, h, d) in occupied:
                x2, y2, z2 = x + w, y + h, z + d
                # Single axis
                candidates.add((x2, y, z))
                candidates.add((x, y2, z))
                candidates.add((x, y, z2))
                # Two axis
                candidates.add((x2, y2, z))
                candidates.add((x2, y, z2))
                candidates.add((x, y2, z2))
                # Three axis
                candidates.add((x2, y2, z2))
                # And combinations with 0
                candidates.add((x2, 0, 0))
                candidates.add((0, y2, 0))
                candidates.add((0, 0, z2))
                candidates.add((x2, y2, 0))
                candidates.add((x2, 0, z2))
                candidates.add((0, y2, z2))

        # Sort by z, then y, then x to prefer Bottom-Left-Back
        return sorted(list(candidates), key=lambda p: (p[2], p[1], p[0]))

    def can_place(x, y, z, w, h, d, occupied):
        """Check if a box at (x,y,z) with size (w,h,d) fits in the container and doesn't overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
            
        for (_, ox, oy, oz, ow, oh, od) in occupied:
            # Check overlap
            if x < ox + ow and x + w > ox and \
               y < oy + oh and y + h > oy and \
               z < oz + od and z + d > oz:
                return False
        return True

    def find_position(box_idx, occupied):
        """Find a valid position for box_idx in a bin with given occupied boxes."""
        w, h, d = boxes[box_idx]
        candidates = get_candidates(occupied)
        
        best_pos = None
        best_key = (float('inf'), float('inf'), float('inf'))
        
        for cx, cy, cz in candidates:
            if can_place(cx, cy, cz, w, h, d, occupied):
                # Heuristic: Minimize z, then y, then x
                key = (cz, cy, cx)
                if key < best_key:
                    best_key = key
                    best_pos = (cx, cy, cz)
        
        return best_pos

    # Initial Packing: First Fit Decreasing
    for idx in indices:
        placed = False
        for i in range(len(bins)):
            pos = find_position(idx, bins[i])
            if pos:
                x, y, z = pos
                w, h, d = boxes[idx]
                bins[i].append((idx, x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Create new bin
            w, h, d = boxes[idx]
            bins.append([(idx, 0, 0, 0, w, h, d)])
            
    # Local Search: Consolidate bins
    # Strategy: Try to move boxes from higher-indexed bins to lower-indexed bins.
    # Also try swapping boxes between bins to free up space.
    
    def get_bin_volume(bin_data):
        return sum(w*h*d for (_, _, _, _, w, h, d) in bin_data)

    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # Try to empty bin i by moving its boxes to bins < i
        for i in range(len(bins) - 1, 0, -1):
            if time.time() > deadline:
                break
            if not bins[i]:
                bins.pop(i)
                improved = True
                continue
                
            # Try to move each box in bin i to a lower bin
            boxes_to_move = list(bins[i])
            for box in boxes_to_move:
                idx = box[0]
                # Try to remove from bin i and place in bin j < i
                moved = False
                # Temporarily remove from bin i to check placement
                bins[i].remove(box)
                
                for j in range(i):
                    if time.time() > deadline:
                        break
                    pos = find_position(idx, bins[j])
                    if pos:
                        x, y, z = pos
                        w, h, d = boxes[idx]
                        bins[j].append((idx, x, y, z, w, h, d))
                        moved = True
                        break
                        
                if not moved:
                    # Put back in bin i
                    bins[i].append(box)
                else:
                    improved = True
                    
            # If bin i is now empty, remove it
            if not bins[i]:
                bins.pop(i)
                improved = True

    # Final Check: Ensure all boxes are placed
    # Reconstruct output in original order
    placements = [None] * n
    
    # Map box index to placement
    for b_idx, bin_data in enumerate(bins):
        for (idx, x, y, z, w, h, d) in bin_data:
            placements[idx] = (b_idx, x, y, z, w, h, d)
            
    # Verify all are placed
    for p in placements:
        if p is None:
            # Fallback: assign to bin 0 at (0,0,0) which will be invalid, but prevents crash.
            # Better: raise error or return empty? The task says "invalid result scores 0".
            # So we must ensure validity.
            pass

    return placements
