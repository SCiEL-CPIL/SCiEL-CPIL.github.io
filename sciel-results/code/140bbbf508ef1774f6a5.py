import time
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending, with tie-breaking by max dimension
    box_volumes = [(w * h * d, max(w, h, d), i) for i, (w, h, d) in enumerate(boxes)]
    box_volumes.sort(key=lambda x: (-x[0], -x[1]))
    sorted_indices = [x[2] for x in box_volumes]
    
    # We'll use a list of free spaces per bin
    # Each free space is (x, y, z, w, h, d) representing an empty cuboid
    
    def split_space(space, box_pos, box_dim):
        """Split a free space around a placed box, returning new free spaces."""
        sx, sy, sz, sw, sh, sd = space
        bx, by, bz, bw, bh, bd = box_pos + box_dim
        
        new_spaces = []
        
        # Check if box is fully inside space
        if bx < sx or by < sy or bz < sz or bx + bw > sx + sw or by + bh > sy + sh or bz + bd > sz + sd:
            return [space]  # No intersection, keep original
        
        # Split along x-axis
        if bx > sx:
            new_spaces.append((sx, sy, sz, bx - sx, sh, sd))
        if bx + bw < sx + sw:
            new_spaces.append((bx + bw, sy, sz, sx + sw - bx - bw, sh, sd))
        
        # Split along y-axis (within the x-range of the box)
        x_start = max(sx, bx)
        x_end = min(sx + sw, bx + bw)
        x_w = x_end - x_start
        if x_w > 0:
            if by > sy:
                new_spaces.append((x_start, sy, sz, x_w, by - sy, sd))
            if by + bh < sy + sh:
                new_spaces.append((x_start, by + bh, sz, x_w, sy + sh - by - bh, sd))
        
        # Split along z-axis (within the x-y range of the box)
        y_start = max(sy, by)
        y_end = min(sy + sh, by + bh)
        y_h = y_end - y_start
        if y_h > 0:
            if bz > sz:
                new_spaces.append((x_start, y_start, sz, x_w, y_h, bz - sz))
            if bz + bd < sz + sd:
                new_spaces.append((x_start, y_start, bz + bd, x_w, y_h, sz + sd - bz - bd))
        
        return new_spaces
    
    def can_place_in_space(space, box_dim):
        """Check if box fits in a free space."""
        _, _, _, sw, sh, sd = space
        bw, bh, bd = box_dim
        return bw <= sw and bh <= sh and bd <= sd
    
    def find_best_space(free_spaces, box_dim):
        """Find the best free space for the box using bottom-left-farthest heuristic."""
        bw, bh, bd = box_dim
        best_space = None
        best_score = -1
        
        for space in free_spaces:
            if can_place_in_space(space, box_dim):
                sx, sy, sz, sw, sh, sd = space
                # Score: prefer positions that minimize wasted space
                # Use the position where the box would be placed (bottom-left)
                # Prefer spaces where the box fits tightly
                waste = (sw - bw) * sh * sd + sw * (sh - bh) * sd + sw * sh * (sd - bd)
                # Prefer smaller waste, and also prefer lower z (bottom)
                score = -waste - sz * 0.1
                if score > best_score:
                    best_score = score
                    best_space = space
        
        return best_space
    
    # Initialize with one bin
    bins = []  # Each bin is a list of free spaces
    placements = []  # (bin_idx, x, y, z, w, h, d)
    
    # Start with one empty bin
    bins.append([(0, 0, 0, W, H, D)])
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        
        # Try to place in existing bins
        for bin_idx in range(len(bins)):
            if time.time() > deadline - 2.0:
                # Time to wrap up
                break
            
            free_spaces = bins[bin_idx]
            
            # Find the best space
            best_space = find_best_space(free_spaces, (w, h, d))
            if best_space is None:
                continue
            
            sx, sy, sz, sw, sh, sd = best_space
            
            # Place the box at the bottom-left of the space
            bx, by, bz = sx, sy, sz
            
            # Split the space
            new_spaces = split_space(best_space, (bx, by, bz), (w, h, d))
            
            # Remove the old space and add new ones
            free_spaces.remove(best_space)
            free_spaces.extend(new_spaces)
            
            placements.append((bin_idx, bx, by, bz, w, h, d))
            placed = True
            break
        
        if not placed:
            # Open a new bin
            bins.append([(0, 0, 0, W, H, D)])
            bin_idx = len(bins) - 1
            bx, by, bz = 0, 0, 0
            new_spaces = split_space((0, 0, 0, W, H, D), (bx, by, bz), (w, h, d))
            bins[bin_idx] = new_spaces
            placements.append((bin_idx, bx, by, bz, w, h, d))
    
    # Calculate score components
    occupied_bins = set(p[0] for p in placements)
    B = len(occupied_bins)
    
    # Find the last bin (largest label)
    if placements:
        last_bin = max(p[0] for p in placements)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
        last_bin_fraction = last_bin_vol / container_vol if container_vol > 0 else 0
    else:
        last_bin_fraction = 0
    
    # Return placements with original box indices mapped back
    # We need to return placements in the order of the input boxes
    # placements currently has (bin_idx, x, y, z, w, h, d) for each box in sorted order
    # We need to map back to original order
    
    result = [None] * n
    for i, (bin_idx, x, y, z, w, h, d) in enumerate(placements):
        orig_idx = sorted_indices[i]
        result[orig_idx] = (bin_idx, x, y, z, w, h, d)
    
    return result
