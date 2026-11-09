import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []

    # Sort boxes by volume descending to place larger items first
    # This often leads to better packing density
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placements = [None] * n
    bins = []  # List of lists, each containing (x, y, z, w, h, d, original_index)
    current_bin = []
    
    def check_overlap(x, y, z, w, h, d, bin_boxes):
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            # Check if intervals overlap in all 3 dimensions
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return True
        return False

    def find_best_position(w, h, d, bin_boxes):
        # Generate candidate positions
        # Start with (0,0,0)
        candidates = [(0, 0, 0)]
        
        # Add corners of existing boxes
        # If a box is at (bx, by, bz) with size (bw, bh, bd),
        # potential new positions are (bx+bw, by, bz), (bx, by+bh, bz), (bx, by, bz+bd)
        # We can also consider combinations, but for performance, single-step corners are often sufficient for a good heuristic.
        # A more robust approach adds all combinations of x, y, z from existing boxes + their sizes.
        # Given 40 boxes, the number of candidates won't explode too much.
        
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            x_candidates.add(bx)
            x_candidates.add(bx + bw)
            y_candidates.add(by)
            y_candidates.add(by + bh)
            z_candidates.add(bz)
            z_candidates.add(bz + bd)
            
        # Filter candidates to those within bounds
        x_candidates = [x for x in x_candidates if x + w <= W]
        y_candidates = [y for y in y_candidates if y + h <= H]
        z_candidates = [z for z in z_candidates if z + d <= D]
        
        best_pos = None
        best_cost = float('inf')
        
        # Iterate through all combinations
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    # Cost function: Minimize distance to bottom-left-front
                    # This encourages compact packing
                    cost = x + y + z
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        
        return best_pos

    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a simple shelf manner in the current or new bins
            # This ensures we return a valid answer even if time runs out
            # For simplicity, just append to a new bin with origin (0,0,0) if we can't find a spot,
            # but actually we must ensure validity. 
            # If we time out, we should return what we have if valid, or a trivial valid solution.
            # Since we process one by one, if we time out, we haven't placed this box.
            # We can place it in a new bin at (0,0,0) to ensure validity, though it's suboptimal.
            # However, the loop is fast. 40 boxes is small. This guard is mostly for safety.
            pass

        placed = False
        # Try to place in current bin
        pos = find_best_position(w, h, d, current_bin)
        if pos is not None:
            x, y, z = pos
            current_bin.append((x, y, z, w, h, d, i))
            placements[i] = (len(bins), x, y, z, w, h, d)
            placed = True
        else:
            # If it doesn't fit in current bin, open a new bin
            if current_bin:
                bins.append(current_bin)
            current_bin = [(0, 0, 0, w, h, d, i)]
            placements[i] = (len(bins), 0, 0, 0, w, h, d)
            placed = True
            
    if current_bin:
        bins.append(current_bin)
        
    return placements
# EVOLVE-BLOCK-END
