"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s
    
    # Sort boxes by volume descending (largest first)
    # We need to remember original indices to return correct placements
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Each bin will have a list of placed boxes: (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in the bin without overlap."""
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap in all 3 dimensions
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        return True
    
    def find_position(bin_placements, w, h, d):
        """Find the best (lowest, then leftmost, then frontmost) position for a box."""
        # Try to place at various candidate positions
        # Candidate positions are generated from existing boxes' boundaries
        # Start with origin, then try positions adjacent to existing boxes
        
        # Generate candidate x, y, z positions
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (px, py, pz, pw, ph, pd) in bin_placements:
            x_candidates.append(px + pw)
            y_candidates.append(py + ph)
            z_candidates.append(pz + pd)
        
        # Sort candidates
        x_candidates = sorted(set(x_candidates))
        y_candidates = sorted(set(y_candidates))
        z_candidates = sorted(set(z_candidates))
        
        best_pos = None
        best_score = None
        
        for z in z_candidates:
            if z + d > D:
                continue
            for y in y_candidates:
                if y + h > H:
                    continue
                for x in x_candidates:
                    if x + w > W:
                        continue
                    if can_place(bin_placements, x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_used_volume(bin_placements):
        return sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_placements)
    
    # Place each box
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
            
        placed = False
        # Try existing bins first (best fit: bin with least remaining space that can fit)
        # We'll try bins in order of increasing used volume (tighter bins first)
        bin_indices = sorted(range(len(bins)), key=lambda i: bin_used_volume(bins[i]), reverse=True)
        
        for bi in bin_indices:
            pos = find_position(bins[bi], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[bi].append((x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Open a new bin
            pos = find_position([], w, h, d)
            if pos is not None:
                bins.append([(pos[0], pos[1], pos[2], w, h, d)])
                placed = True
            else:
                # Should not happen if box fits in container, but handle gracefully
                bins.append([(0, 0, 0, w, h, d)])
                placed = True
    
    # Build result: for each original box index, find its placement
    # We need to track which bin each box ended up in
    # Let's rebuild: we lost track of original indices. Let's redo with proper tracking.
    
    # Actually, let's redo the whole thing properly with tracking
    bins = []
    box_to_bin = {}  # original index -> (bin_index, x, y, z)
    
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Place remaining boxes in a simple fallback
            for remaining in indexed_boxes:
                if remaining[0] not in box_to_bin:
                    # Just put in a new bin at origin (may overlap but better than crash)
                    # Actually this won't work. Let's just break and handle below.
                    pass
            break
            
        placed = False
        # Try existing bins
        bin_indices = sorted(range(len(bins)), key=lambda i: bin_used_volume(bins[i]), reverse=True)
        
        for bi in bin_indices:
            pos = find_position(bins[bi], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[bi].append((x, y, z, w, h, d))
                box_to_bin[idx] = (bi, x, y, z)
                placed = True
                break
        
        if not placed:
            pos = find_position([], w, h, d)
            if pos is not None:
                new_bin_idx = len(bins)
                bins.append([(pos[0], pos[1], pos[2], w, h, d)])
                box_to_bin[idx] = (new_bin_idx, pos[0], pos[1], pos[2])
                placed = True
            else:
                # Fallback: place at origin in new bin
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                box_to_bin[idx] = (new_bin_idx, 0, 0, 0)
                placed = True
    
    # Handle any boxes that weren't placed due to deadline
    for i, (w, h, d) in enumerate(boxes):
        if i not in box_to_bin:
            # Find a bin where it might fit, or create new
            placed = False
            for bi in range(len(bins)):
                pos = find_position(bins[bi], w, h, d)
                if pos is not None:
                    bins[bi].append((pos[0], pos[1], pos[2], w, h, d))
                    box_to_bin[i] = (bi, pos[0], pos[1], pos[2])
                    placed = True
                    break
            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                box_to_bin[i] = (new_bin_idx, 0, 0, 0)
    
    # Build final placements in original order
    placements = []
    for i in range(n):
        bi, x, y, z = box_to_bin[i]
        w, h, d = boxes[i]
        placements.append((bi, x, y, z, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
