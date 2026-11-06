"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    Place boxes into bins using a Best-Fit-Decreasing heuristic with local search compaction.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Safety margin: The evaluator kills at 120s (or 150s per prompt text, but 120s is safer).
    # Let's aim to finish well before 115s.
    deadline = time.time() + 110.0 

    # 1. Sort boxes by volume descending (Largest First)
    # We keep original indices to restore order at the end
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # bins: list of lists. Each inner list contains (x, y, z, w, h, d) of placed boxes.
    bins = []
    
    # Mapping from original index to its placement (bin_id, x, y, z, w, h, d)
    # We will store this in a dict or list, but since we move boxes during local search,
    # it's better to reconstruct the final result from the `bins` structure at the end.
    # However, we need to know which box is which. 
    # Let's store boxes as objects or tuples with an ID.
    # Box representation in bins: (id, x, y, z, w, h, d)
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) in the given bin without overlap and within bounds."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (pid, px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def find_best_position(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin.
        Strategy: Generate candidate positions from the "corners" of existing boxes.
        Score: Minimize the sum of coordinates (prefer bottom-left-front).
        """
        if not bin_placements:
            # If bin is empty, place at origin
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None

        # Generate candidate positions
        # The "free space" corners are formed by the extents of existing boxes.
        # We consider x-coordinates from {0} U {px + pw for each box}
        # Same for y and z.
        # This creates a grid of potential positions.
        
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (pid, px, py, pz, pw, ph, pd) in bin_placements:
            if px + pw <= W: xs.add(px + pw)
            if py + ph <= H: ys.add(py + ph)
            if pz + pd <= D: zs.add(pz + pd)
            
        # Sort to ensure deterministic behavior and potentially find "lowest" positions earlier
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        best_pos = None
        best_score = float('inf')
        
        # Iterate through candidate positions
        # To optimize, we can skip positions that are obviously bad, but with N=40, 
        # the number of candidates is manageable (approx (N+1)^3 in worst case, but usually less).
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    
                    if can_place(bin_placements, x, y, z, w, h, d):
                        # Score: Prefer positions that are "lower" in the bin.
                        # This encourages stacking and filling from the bottom up.
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            
        return best_pos

    def bin_bounding_box_volume(bin_placements):
        """Calculate the volume of the bounding box of the bin's contents."""
        if not bin_placements:
            return 0
        max_x = max(px + pw for (pid, px, py, pz, pw, ph, pd) in bin_placements)
        max_y = max(py + ph for (pid, px, py, pz, pw, ph, pd) in bin_placements)
        max_z = max(pz + pd for (pid, px, py, pz, pw, ph, pd) in bin_placements)
        return max_x * max_y * max_z

    # --- Phase 1: Greedy Best-Fit Decreasing ---
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline - 5:
            # If we are running out of time, place remaining boxes in new bins or first available
            # to ensure we return *something* valid
            pass
            
        best_bin_idx = -1
        best_pos = None
        best_cost = float('inf')
        
        # Try to place in existing bins
        for b_idx, bin_placements in enumerate(bins):
            pos = find_best_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Cost: We want to minimize the "wasted space" or keep the bounding box tight.
                # Let's use the new bounding box volume as a proxy.
                # Calculate new max extents
                new_max_x = max(x + w, *(px + pw for (pid, px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else x + w
                new_max_y = max(y + h, *(py + ph for (pid, px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else y + h
                new_max_z = max(z + d, *(pz + pd for (pid, px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else z + d
                
                cost = new_max_x * new_max_y * new_max_z
                
                if cost < best_cost:
                    best_cost = cost
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx >= 0:
            # Place in the best existing bin
            x, y, z = best_pos
            bins[best_bin_idx].append((idx, x, y, z, w, h, d))
        else:
            # Create a new bin
            # Check if it fits in a new bin (it should, assuming boxes are smaller than container)
            if w <= W and h <= H and d <= D:
                new_bin = [(idx, 0, 0, 0, w, h, d)]
                bins.append(new_bin)
            else:
                # Box is larger than container? This shouldn't happen in valid inputs, 
                # but if it does, we still need to place it somewhere to return a result.
                # The problem states "places every box... into numbered bins".
                # If it doesn't fit, the score is likely invalid or 0, but we must return a placement.
                # We'll place it at 0,0,0 in a new bin anyway.
                new_bin = [(idx, 0, 0, 0, w, h, d)]
                bins.append(new_bin)

    # --- Phase 2: Local Search / Compaction ---
    # Try to move boxes from later bins to earlier bins to reduce the total number of bins.
    # We iterate from the last bin backwards.
    
    # To speed up local search, we can limit the number of passes or time spent.
    # Since N=40, the state space is small enough for a few passes.
    
    compaction_passed = True
    passes = 0
    max_passes = 10 # Safety limit
    
    while compaction_passed and passes < max_passes and time.time() < deadline - 5:
        compaction_passed = False
        passes += 1
        
        # Iterate bins from last to first
        for b_idx in range(len(bins) - 1, 0, -1):
            if time.time() > deadline - 5:
                break
                
            if not bins[b_idx]:
                # If bin is empty, remove it (shouldn't happen if we manage removals correctly)
                bins.pop(b_idx)
                compaction_passed = True
                continue
                
            # Try to move each box in bin b_idx to an earlier bin
            # We must be careful with iteration if we modify the list.
            # We'll create a list of boxes to try and process them one by one.
            boxes_in_bin = list(bins[b_idx])
            
            for box in boxes_in_bin:
                if time.time() > deadline - 5:
                    break
                    
                box_id, bx, by, bz, bw, bh, bd = box
                
                # Try to find a spot in bins [0, b_idx)
                moved = False
                for target_bin_idx in range(b_idx):
                    # Check if box is still in the source bin (it might have been moved in a previous iteration of this loop? No, we break on move)
                    # We need to ensure we are checking against the current state of the target bin.
                    
                    # Optimization: Quick check if box fits in target bin's bounding box? 
                    # find_best_position does the detailed check.
                    
                    pos = find_best_position(bins[target_bin_idx], bw, bh, bd)
                    if pos is not None:
                        # Move the box
                        # Remove from source bin
                        bins[b_idx].remove(box)
                        # Add to target bin
                        nx, ny, nz = pos
                        bins[target_bin_idx].append((box_id, nx, ny, nz, bw, bh, bd))
                        moved = True
                        compaction_passed = True
                        break
                        
                if moved:
                    # If the source bin is now empty, we can remove it
                    # But careful: b_idx index changes if we remove.
                    # We are iterating b_idx downwards. If we remove bins[b_idx], 
                    # the loop continues to b_idx-1 which is now at index b_idx-1 in the list.
                    # However, the `for b_idx in range(...)` is fixed at start.
                    # So if we remove an element, the indices shift.
                    # It's safer to NOT remove empty bins inside the inner loop, 
                    # but clean them up after a full pass or at the end.
                    # Or, we can just leave them empty and handle in final cleanup.
                    # Let's leave them and clean up at the end of the while loop.
                    pass

        # Clean up empty bins after a full pass
        if compaction_passed:
            new_bins = []
            for b in bins:
                if b:
                    new_bins.append(b)
            bins = new_bins

    # --- Phase 3: Finalize Result ---
    
    # We need to return a list of (bin_id, x, y, z, w, h, d) for each original box index.
    # bin_id should be the index in the `bins` list.
    
    result = [None] * n
    
    for b_idx, bin_placements in enumerate(bins):
        for (box_id, x, y, z, w, h, d) in bin_placements:
            result[box_id] = (b_idx, x, y, z, w, h, d)
            
    # Fallback: If any box is not placed (should not happen), place it in a new bin at 0,0,0
    current_max_bin = len(bins)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (current_max_bin, 0, 0, 0, w, h, d)
            current_max_bin += 1

    return result
# EVOLVE-BLOCK-END
