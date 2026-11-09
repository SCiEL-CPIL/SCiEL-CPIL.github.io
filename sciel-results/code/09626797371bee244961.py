import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill
    
    W, H, D = container
    # Ensure dimensions are integers for grid indexing
    W = int(W)
    H = int(H)
    D = int(D)
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    # We keep track of original indices to return correct placements
    indexed = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Each bin is represented by:
    # 1. A list of placed boxes: (orig_idx, x, y, z, w, h, d)
    # 2. A 3D boolean occupancy grid: np.zeros((W, H, D), dtype=bool)
    bins = []
    
    def create_bin():
        return {
            'boxes': [],
            'grid': np.zeros((W, H, D), dtype=bool)
        }

    def can_place(grid, x, y, z, w, h, d):
        """Check if a box of size (w,h,d) fits at (x,y,z) in the given grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any part of the target volume is already occupied
        if np.any(grid[x:x+w, y:y+h, z:z+d]):
            return False
        return True

    def place_box(bin_data, w, h, d):
        """
        Find the best position for a box in the given bin.
        Returns (x, y, z) if found, else None.
        Uses a heuristic: minimize x + y + z to pack towards origin.
        """
        best_pos = None
        best_score = float('inf')
        
        # Iterate through all possible positions
        # Optimization: We can break early if we find a position with score 0 (origin)
        for x in range(W - w + 1):
            for y in range(H - h + 1):
                for z in range(D - d + 1):
                    if can_place(bin_data['grid'], x, y, z, w, h, d):
                        # Heuristic: Prefer positions closer to the origin (bottom-left-front)
                        # This tends to reduce fragmentation by filling corners first.
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            if best_score == 0:
                                return best_pos
        return best_pos

    def mark_grid(grid, x, y, z, w, h, d, value=True):
        grid[x:x+w, y:y+h, z:z+d] = value

    # Initial Placement Phase
    placements = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed:
        if time.time() > DEADLINE:
            break
            
        w = int(w)
        h = int(h)
        d = int(d)
        
        # Ensure box dimensions fit in container
        if w > W or h > H or d > D:
            # This box cannot fit in any bin. 
            # The problem statement implies all boxes fit, but just in case:
            # We must place it. If it doesn't fit, the solution is invalid.
            # Assuming valid input where boxes fit.
            pass

        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        # Try to place in existing bins
        for bi, bin_data in enumerate(bins):
            pos = place_box(bin_data, w, h, d)
            if pos is not None:
                # Calculate a score for this placement to choose the best bin
                # We want to minimize the "wasted space" or maximize density.
                # Simple proxy: Prefer bins where the box is placed lower (smaller z) 
                # and prefer bins that are already fuller (hard to compute quickly).
                # For now, just take the first valid bin? No, Best-Fit means 
                # we should prefer the bin where the box fits "tightest".
                # A common heuristic is to minimize the maximum coordinate reached.
                # Or simply: try to fill bins in order? 
                # Actually, standard BFD tries to find the position that minimizes 
                # the distance to the "floor" or other boxes. 
                # Let's stick to: Find the position in each bin that minimizes x+y+z.
                # Then among bins, choose the one where the box is placed at the lowest z?
                # Or choose the bin that results in the smallest "bounding box" increase?
                
                # Let's use a simple scoring: 
                # 1. If we can place it at z=0, that's good.
                # 2. Among valid positions in a bin, we already picked the one with min x+y+z.
                # 3. To choose between bins, we can prefer the bin that is "fuller" 
                #    to avoid opening new bins prematurely if possible, but BFD usually 
                #    just takes the first bin where it fits if we process bins in order?
                #    No, BFD checks all bins. 
                #    Let's score by the z-coordinate of the placement. Lower z is better.
                #    If z is same, lower y, then lower x.
                
                cx, cy, cz = pos
                # Score: primarily minimize z (height), then y, then x
                bin_score = cz * 10000 + cy * 100 + cx
                
                if bin_score < best_score:
                    best_score = bin_score
                    best_bin_idx = bi
                    best_pos = pos

        if best_bin_idx == -1:
            # No existing bin works. Create a new one.
            new_bin = create_bin()
            # Place at origin
            pos = (0, 0, 0)
            if can_place(new_bin['grid'], 0, 0, 0, w, h, d):
                bins.append(new_bin)
                best_bin_idx = len(bins) - 1
                best_pos = pos
            else:
                # Should not happen if box fits in container
                continue

        # Commit the placement
        x, y, z = best_pos
        mark_grid(bins[best_bin_idx]['grid'], x, y, z, w, h, d)
        bins[best_bin_idx]['boxes'].append((orig_idx, x, y, z, w, h, d))
        placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)

    # Handle any boxes not placed due to time or error (should be none)
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            # Force place in a new bin at origin
            new_bin = create_bin()
            bins.append(new_bin)
            bi = len(bins) - 1
            placements[i] = (bi, 0, 0, 0, int(w), int(h), int(d))
            mark_grid(bins[bi]['grid'], 0, 0, 0, int(w), int(h), int(d))
            bins[bi]['boxes'].append((i, 0, 0, 0, int(w), int(h), int(d)))

    # Local Improvement: Try to move boxes from higher bins to lower bins
    # This is a "compaction" step.
    if time.time() < DEADLINE - 10:
        max_bins = max(p[0] for p in placements) if placements else 0
        
        # Repeat passes until no improvement or time limit
        for _ in range(20):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            # Iterate through boxes in reverse order of placement (largest last?) 
            # Or just iterate all boxes.
            for i in range(len(boxes)):
                bi, x, y, z, w, h, d = placements[i]
                
                # Try to move to any lower bin
                for target_bin in range(bi - 1, -1, -1):
                    # Remove from current bin
                    current_bin = bins[bi]
                    # Find and remove the box from the list and grid
                    # Note: We store (orig_idx, x, y, z, w, h, d) in bin['boxes']
                    # We need to remove the specific entry.
                    # Since we know the position and size, we can unmark the grid.
                    mark_grid(current_bin['grid'], x, y, z, w, h, d, False)
                    
                    # Find the box in the list to remove it
                    # This is O(N) but N is small (40 boxes)
                    box_entry = (i, x, y, z, w, h, d)
                    if box_entry in current_bin['boxes']:
                        current_bin['boxes'].remove(box_entry)
                    
                    # Try to place in target bin
                    target_bin_data = bins[target_bin]
                    pos = place_box(target_bin_data, w, h, d)
                    
                    if pos is not None:
                        # Success! Move the box
                        tx, ty, tz = pos
                        mark_grid(target_bin_data['grid'], tx, ty, tz, w, h, d, True)
                        target_bin_data['boxes'].append((i, tx, ty, tz, w, h, d))
                        placements[i] = (target_bin, tx, ty, tz, w, h, d)
                        improved = True
                        break # Move to next box
                    else:
                        # Failed, restore to original bin
                        mark_grid(current_bin['grid'], x, y, z, w, h, d, True)
                        current_bin['boxes'].append((i, x, y, z, w, h, d))
                        # If it didn't fit in this lower bin, it won't fit in any higher bin 
                        # that is "worse" (but we are checking lower bins). 
                        # Actually, if it doesn't fit in bin k, it might fit in bin k+1? 
                        # No, we are trying to move FROM bi TO lower. 
                        # If it doesn't fit in target_bin, we restore and try next lower target?
                        # Or just break? 
                        # If it doesn't fit in the immediate lower bin, it might fit in a lower one?
                        # Unlikely if bins are packed tightly, but possible. 
                        # Let's try all lower bins.
                        continue

            if not improved:
                break

    return placements
