"""3D Bin Packing with Best-Fit Decreasing + local improvement."""
import time
import sys
import math
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    def split_space(sx, sy, sz, sw, sh, sd, bx, by, bz, bw, bh, bd):
        """Split space (sx,sy,sz,sw,sh,sd) by box (bx,by,bz,bw,bh,bd).
        Returns list of up to 6 new non-overlapping sub-spaces."""
        result = []
        # Left side
        if bx > sx:
            result.append((sx, sy, sz, bx - sx, sh, sd))
        # Right side
        if bx + bw < sx + sw:
            result.append((bx + bw, sy, sz, sx + sw - bx - bw, sh, sd))
        # Back side (lower y)
        if by > sy:
            result.append((sx, sy, sz, sw, by - sy, sd))
        # Front side (higher y)
        if by + bh < sy + sh:
            result.append((sx, by + bh, sz, sw, sy + sh - by - bh, sd))
        # Bottom side (lower z)
        if bz > sz:
            result.append((sx, sy, sz, sw, sh, bz - sz))
        # Top side (higher z)
        if bz + bd < sz + sd:
            result.append((sx, sy, bz + bd, sw, sh, sz + sd - bz - bd))
        return result
    
    def find_position(free_spaces, w, h, d, W, H, D):
        """Find best position using free-space list. Returns (x,y,z) or None."""
        best_pos = None
        best_score = float('inf')
        
        for (sx, sy, sz, sw, sh, sd) in free_spaces:
            if w > sw or h > sh or d > sd:
                continue
            # Box fits in this free space at position (sx, sy, sz)
            # Score: prefer lower z, then lower y, then lower x
            score = sz * 10000 + sy * 100 + sx
            if score < best_score:
                best_score = score
                best_pos = (sx, sy, sz)
        
        return best_pos
    
    def place_in_bin(free_spaces, x, y, z, w, h, d):
        """Update free-space list after placing a box at (x,y,z) with size (w,h,d)."""
        new_free = []
        for (sx, sy, sz, sw, sh, sd) in free_spaces:
            # Check if box intersects this free space
            if x + w <= sx or sx + sw <= x or y + h <= sy or sy + sh <= y or z + d <= sz or sz + sd <= z:
                # No intersection, keep as is
                new_free.append((sx, sy, sz, sw, sh, sd))
            else:
                # Split this free space around the box
                pieces = split_space(sx, sy, sz, sw, sh, sd, x, y, z, w, h, d)
                new_free.extend(pieces)
        return new_free
    
    def run_bfd(sorted_indices):
        """Run Best-Fit Decreasing with free-space representation.
        Returns list of (bin, x, y, z, w, h, d)."""
        # Each bin has a list of free spaces, starting with the full container
        bin_free = []  # list of lists of (sx, sy, sz, sw, sh, sd)
        placements = [None] * n  # indexed by original box index
        
        for idx in sorted_indices:
            orig_idx, w, h, d = indexed_boxes[idx]
            
            best_bin = -1
            best_score = float('inf')
            best_pos = None
            
            for b in range(len(bin_free)):
                pos = find_position(bin_free[b], w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    # Score: prefer lower z, then lower y, then lower x
                    score = z * 10000 + y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_bin = b
                        best_pos = pos
            
            if best_bin == -1:
                # Open a new bin
                bin_free.append([(0, 0, 0, W, H, D)])
                best_bin = len(bin_free) - 1
                best_pos = (0, 0, 0)
            
            x, y, z = best_pos
            bin_free[best_bin] = place_in_bin(bin_free[best_bin], x, y, z, w, h, d)
            placements[orig_idx] = (best_bin, x, y, z, w, h, d)
        
        return placements, bin_free
    
    def compute_score(placements):
        if not placements:
            return 0.0
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(bin_labels)
        last_bin_vol = 0
        for p in placements:
            if p[0] == max_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        container_vol = W * H * D
        if container_vol > 0:
            frac = last_bin_vol / container_vol
        else:
            frac = 0.0
        return B + 0.1 * (1 - frac)
    
    # Run initial BFD
    initial_order = list(range(n))
    placements, _ = run_bfd(initial_order)
    
    best_placements = placements
    best_score = compute_score(placements)
    
    elapsed = time.time() - start_time
    remaining = deadline - start_time - 5.0
    
    if remaining <= 0:
        return best_placements
    
    # Local search: try different orderings via pairwise swaps
    max_iterations = 500
    iteration = 0
    current_order = initial_order[:]
    
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        # Pick two random indices to swap
        i, j = random.sample(range(n), 2)
        
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_placements, _ = run_bfd(new_order)
        new_score = compute_score(new_placements)
        
        if new_score < best_score:
            best_score = new_score
            best_placements = new_placements
            current_order = new_order
        
        iteration += 1
    
    return best_placements
