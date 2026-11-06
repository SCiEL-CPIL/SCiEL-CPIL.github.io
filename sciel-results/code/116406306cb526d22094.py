"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # Maximal Spaces representation: (x, y, z, w, h, d)
    # Initially, the whole container is one maximal space
    max_spaces = [(0, 0, 0, W, H, D)]
    
    placements = [None] * len(boxes)
    
    def is_contained(s1, s2):
        """Check if space s1 is contained within space s2."""
        x1, y1, z1, w1, h1, d1 = s1
        x2, y2, z2, w2, h2, d2 = s2
        return (x2 <= x1 and y2 <= y1 and z2 <= z1 and 
                x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1 and z2 + d2 >= z1 + d1)
    
    def prune_spaces(spaces):
        """Remove spaces that are contained in other spaces."""
        n = len(spaces)
        if n <= 1:
            return spaces
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if is_contained(spaces[i], spaces[j]):
                    keep[i] = False
                    break
        return [s for i, s in enumerate(spaces) if keep[i]]
    
    def place_in_space(space, w, h, d):
        """
        Try to place a box of size (w,h,d) in the given maximal space.
        Returns (x, y, z, new_spaces) if successful, else None.
        We try to place it at the bottom-left-front of the space to keep packing tight.
        """
        sx, sy, sz, sw, sh, sd = space
        if w > sw or h > sh or d > sd:
            return None
        
        # Place at the start of the space
        px, py, pz = sx, sy, sz
        
        # Calculate new spaces after placing the box
        new_spaces = []
        
        # 1. Space above the box (if any)
        if py + h < sy + sh:
            new_spaces.append((sx, py + h, sz, sw, sy + sh - (py + h), sd))
        # 2. Space to the right of the box (if any)
        if px + w < sx + sw:
            new_spaces.append((px + w, sy, sz, sx + sw - (px + w), sh, sd))
        # 3. Space behind the box (if any)
        if pz + d < sz + sd:
            new_spaces.append((sx, sy, pz + d, sw, sh, sz + sd - (pz + d)))
            
        return (px, py, pz, new_spaces)

    def try_place_box(box_idx, bw, bh, bd):
        """
        Try to place the box with original dimensions (bw, bh, bd) in any rotation.
        Returns True if placed, False otherwise.
        """
        for perm in permutations([bw, bh, bd]):
            pw, ph, pd = perm
            
            # Iterate through a copy of max_spaces because we might modify it
            # We want to find the first space that fits
            # To improve quality, we might sort spaces by size or position, 
            # but simple iteration is standard. 
            # Heuristic: Try spaces that are "tighter" or smaller first to avoid wasting large spaces?
            # Actually, standard MS tries all and picks the first fit. 
            # Let's try to find the "best" fit among all spaces for this rotation.
            # Best fit could be defined as the space with the smallest remaining volume after placement,
            # or the one closest to origin.
            
            best_fit = None
            best_score = float('inf')
            
            for space in max_spaces:
                res = place_in_space(space, pw, ph, pd)
                if res:
                    px, py, pz, new_spaces = res
                    # Score: prefer positions closer to origin (0,0,0)
                    # This tends to keep the packing dense.
                    score = px + py + pz
                    if score < best_score:
                        best_score = score
                        best_fit = (space, px, py, pz, new_spaces)
            
            if best_fit:
                space, px, py, pz, new_spaces = best_fit
                # Remove the used space from max_spaces
                max_spaces.remove(space)
                # Add new spaces
                max_spaces.extend(new_spaces)
                # Prune redundant spaces
                max_spaces = prune_spaces(max_spaces)
                
                placements[box_idx] = (0, px, py, pz, pw, ph, pd)
                return True
        
        return False

    for box_idx, (w, h, d) in indexed_boxes:
        if not try_place_box(box_idx, w, h, d):
            placements[box_idx] = (-1, 0, 0, 0, w, h, d)
    
    # Ensure all placements are set
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
