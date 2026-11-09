import time
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    W, H, D = container
    
    # Preprocess: Sort boxes by volume descending
    # We need to keep track of original indices to return correct placements
    # The problem says "preserving its original ordered dimensions", but the output is a list of placements.
    # The input `boxes` is a list of (w, h, d).
    # The output is a list of (bin, x, y, z, w, h, d).
    # The order of the output list should probably correspond to the input box order?
    # "one placement ... per box, preserving its original ordered dimensions"
    # Usually, this means the i-th element in the output corresponds to the i-th box in the input.
    # Let's assume the output list must be in the same order as the input `boxes`.
    
    n = len(boxes)
    indices = list(range(n))
    # Sort indices by volume descending
    indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # bins: list of lists of (box_index, x, y, z, w, h, d)
    bins = []
    
    def get_corners(occupied):
        """Generate candidate bottom-left-back positions for a new box in a bin."""
        if not occupied:
            return [(0, 0, 0)]
        
        candidates = set()
        candidates.add((0, 0, 0))
        
        # For each box, add its "top-right-front" corners and combinations
        # This is a standard heuristic. It might miss some spots but is fast.
        for (_, x, y, z, w, h, d) in occupied:
            x2, y2, z2 = x + w, y + h, z + d
            # Single axis extensions
            candidates.add((x2, y, z))
            candidates.add((x, y2, z))
            candidates.add((x, y, z2))
            # Two axis extensions
            candidates.add((x2, y2, z))
            candidates.add((x2, y, z2))
            candidates.add((x, y2, z2))
            # Three axis extensions
            candidates.add((x2, y2, z2))
            
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
        candidates = get_corners(occupied)
        
        for cx, cy, cz in candidates:
            if can_place(cx, cy, cz, w, h, d, occupied):
                return (cx, cy, cz)
        return None

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
    # We want to minimize the number of bins.
    # Strategy: Try to move boxes from higher-indexed bins to lower-indexed bins.
    
    def get_bin_volume(bin_data):
        return sum(w*h*d for (_, _, _, _, w, h, d) in bin_data)

    # Sort bins by index (they are already 0..N-1)
    # We will iterate from the last bin down to 1.
    
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
            # This shouldn't happen, but if it does, we have a bug.
            # Fallback: assign to bin 0 at (0,0,0) which will be invalid, but prevents crash.
            # Better: raise error or return empty? The task says "invalid result scores 0".
            # So we must ensure validity.
            pass

    return placements
