import time
import math
import random
from typing import List, Tuple

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: Evaluator kills at 150s, but task says 120s. 
    # Use 115s to be safe.
    deadline = time.time() + 115.0
    
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    min_bins = max(1, math.ceil(total_vol / container_vol))
    
    # Sort indices by volume descending (Largest First)
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)
    
    def compute_score(placements, W, H, D):
        if not placements:
            return 0.0
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_bin_vol = sum(p[3] * p[4] * p[5] for p in placements if p[0] == last_bin)
        return B + 0.1 * (1 - last_bin_vol / container_vol)

    def split_space(space, box):
        """
        Split a free space by placing a box at (bx, by, bz) with dimensions (bw, bh, bd).
        Returns a list of up to 6 new non-overlapping sub-spaces.
        space = (sx, sy, sz, sw, sh, sd)
        box = (bx, by, bz, bw, bh, bd)
        """
        sx, sy, sz, sw, sh, sd = space
        bx, by, bz, bw, bh, bd = box
        
        new_spaces = []
        
        # Check if box is fully contained in space
        if bx < sx or by < sy or bz < sz:
            return []
        if bx + bw > sx + sw or by + bh > sy + sh or bz + bd > sz + sd:
            return []
        
        # Left
        if bx > sx:
            new_spaces.append((sx, sy, sz, bx - sx, sh, sd))
        # Right
        if bx + bw < sx + sw:
            new_spaces.append((bx + bw, sy, sz, sx + sw - bx - bw, sh, sd))
        # Bottom
        if by > sy:
            new_spaces.append((sx, sy, sz, sw, by - sy, sd))
        # Top
        if by + bh < sy + sh:
            new_spaces.append((sx, by + bh, sz, sw, sy + sh - by - bh, sd))
        # Back
        if bz > sz:
            new_spaces.append((sx, sy, sz, sw, sh, bz - sz))
        # Front
        if bz + bd < sz + sd:
            new_spaces.append((sx, sy, bz + bd, sw, sh, sz + sd - bz - bd))
        
        return new_spaces

    def remove_contained_spaces(spaces, new_space):
        """
        Remove any spaces that are fully contained in new_space.
        """
        nsx, nsy, nsz, nsw, nsh, nsd = new_space
        result = []
        for s in spaces:
            sx, sy, sz, sw, sh, sd = s
            # Check if s is fully contained in new_space
            if (sx >= nsx and sy >= nsy and sz >= nsz and
                sx + sw <= nsx + nsw and sy + sh <= nsy + nsh and sz + sd <= nsz + nsd):
                continue
            result.append(s)
        return result

    def update_spaces(spaces, box, space_idx):
        """
        Update the list of free spaces after placing a box in a specific space.
        box = (bx, by, bz, bw, bh, bd)
        space_idx is the index of the space in the list where the box was placed.
        """
        # The box is placed at the bottom-left-back of the space at space_idx.
        # We remove that space and replace it with up to 3 new disjoint spaces.
        
        sx, sy, sz, sw, sh, sd = spaces[space_idx]
        bx, by, bz, bw, bh, bd = box
        
        new_spaces = []
        
        # 1. Right slab: x from bx+bw to sx+sw, y from sy to sy+sh, z from sz to sz+sd
        if bx + bw < sx + sw:
            new_spaces.append((bx + bw, sy, sz, sx + sw - (bx + bw), sh, sd))
            
        # 2. Top slab: x from bx to bx+bw, y from by+bh to sy+sh, z from sz to sz+sd
        if by + bh < sy + sh:
            new_spaces.append((bx, by + bh, sz, bw, sy + sh - (by + bh), sd))
            
        # 3. Back slab: x from bx to bx+bw, y from by to by+bh, z from bz+bd to sz+sd
        if bz + bd < sz + sd:
            new_spaces.append((bx, by, bz + bd, bw, bh, sz + sd - (bz + bd)))
            
        # Reconstruct the list: keep all spaces except the one at space_idx, add new ones
        # It's safer to build a new list
        final_spaces = []
        for i, s in enumerate(spaces):
            if i != space_idx:
                final_spaces.append(s)
        final_spaces.extend(new_spaces)
        
        return final_spaces

    def find_position_in_bin(spaces, w, h, d, W, H, D):
        """
        Find the best position for a box (w,h,d) in a bin with given free spaces.
        Returns (x, y, z, space_index) or None if no position fits.
        """
        best_pos = None
        best_score = float('inf')
        best_idx = -1
        
        for i, (sx, sy, sz, sw, sh, sd) in enumerate(spaces):
            if sw < w or sh < h or sd < d:
                continue
            
            # Try to place at the "bottom-left-back" of this space
            # This tends to pack well
            cx, cy, cz = sx, sy, sz
            
            # Score: minimize z, then y, then x (bottom-left-back)
            # Also prefer larger spaces? No, standard BFD is fine.
            score = cz * 10000 + cy * 100 + cx
            
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
                best_idx = i
        
        if best_pos is None:
            return None
        return (best_pos[0], best_pos[1], best_pos[2], best_idx)

    def solve(ordering):
        """
        Run the packing algorithm with a specific ordering of box indices.
        Returns a list of placements or None if it fails.
        """
        # state: bin_id -> list of free spaces (sx, sy, sz, sw, sh, sd)
        state = {}
        placements = [None] * len(ordering)
        
        for i, box_idx in enumerate(ordering):
            if time.time() > deadline:
                return None
            
            w, h, d = boxes[box_idx]
            
            best_bin = None
            best_pos = None
            best_score = float('inf')
            
            # Try existing bins
            for bin_id in sorted(state.keys()):
                res = find_position_in_bin(state[bin_id], w, h, d, W, H, D)
                if res is None:
                    continue
                
                pos_x, pos_y, pos_z, space_idx = res
                
                # Score this bin+position
                # Prefer bins that are already "fuller" (less remaining volume)
                remaining_vol = sum(sw * sh * sd for (sx, sy, sz, sw, sh, sd) in state[bin_id])
                s = remaining_vol * 0.001 + pos_z * 100 + pos_y * 10 + pos_x
                
                if s < best_score:
                    best_score = s
                    best_bin = bin_id
                    best_pos = (pos_x, pos_y, pos_z)
                    best_space_idx = space_idx
                        
            # If no existing bin works, open a new one
            if best_bin is None:
                new_bin_id = max(state.keys()) + 1 if state else 0
                best_bin = new_bin_id
                best_pos = (0, 0, 0)
                best_space_idx = 0
                state[best_bin] = [(0, 0, 0, W, H, D)]
            
            # Place the box and update spaces
            box_tuple = (best_pos[0], best_pos[1], best_pos[2], w, h, d)
            state[best_bin] = update_spaces(state[best_bin], box_tuple, best_space_idx)
            
            placements[i] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            
        # Reorder placements to match original box index order
        final_placements = [None] * n
        for i, box_idx in enumerate(ordering):
            final_placements[box_idx] = placements[i]
            
        return final_placements

    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Volume Descending (Standard BFD)
    result = solve(indices_sorted)
    if result:
        s = compute_score(result, W, H, D)
        if s < best_score:
            best_score = s
            best_result = result
    
    # Strategy 2: Volume Descending, but tie-break by max dimension
    # This can help align boxes better
    indices_sorted2 = sorted(indices, key=lambda i: (box_vols[i], max(boxes[i])), reverse=True)
    if time.time() < deadline:
        result = solve(indices_sorted2)
        if result:
            s = compute_score(result, W, H, D)
            if s < best_score:
                best_score = s
                best_result = result

    # Strategy 3: Randomized Restart with small perturbation
    # If we have time, try a few random shuffles of the last few items or slight reordering
    # This helps escape local minima where a specific order blocks a better fit.
    if time.time() < deadline - 10: # Leave 10s for safety
        attempts = 0
        while time.time() < deadline - 5 and attempts < 5:
            # Create a slightly perturbed ordering
            # Swap a few random adjacent items in the sorted list
            perturbed = indices_sorted[:]
            # Randomly swap 2-5 pairs
            for _ in range(random.randint(2, 5)):
                i = random.randint(0, len(perturbed) - 2)
                perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
            
            result = solve(perturbed)
            if result:
                s = compute_score(result, W, H, D)
                if s < best_score:
                    best_score = s
                    best_result = result
            attempts += 1

    # Fallback if something went wrong (should not happen)
    if best_result is None:
        # Simple shelf as absolute last resort
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements

    return best_result
