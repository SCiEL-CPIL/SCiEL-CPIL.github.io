"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume in descending order
    # We keep track of original indices
    sorted_indices = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # List of free rectangular prisms: (x, y, z, w, h, d)
    # Initially, the whole container is free
    free_prisms = [(0, 0, 0, W, H, D)]
    
    placements = [None] * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin
    
    for idx in sorted_indices:
        if time.time() > deadline:
            # If time runs out, mark remaining as left out
            # This is a safety measure
            for remaining_idx in range(idx, n):
                if placements[remaining_idx] is None:
                    w, h, d = boxes[remaining_idx]
                    placements[remaining_idx] = (-1, 0, 0, 0, w, h, d)
            break
        
        w, h, d = boxes[idx]
        best_score = float('inf')
        best_prism_idx = -1
        best_rot = None
        best_pos = None
        
        # Generate unique rotations to avoid redundant checks
        # Standard 6 permutations
        rotations = [
            (w, h, d), (w, d, h),
            (h, w, d), (h, d, w),
            (d, w, h), (d, h, w)
        ]
        
        # Iterate through all free prisms to find the best fit
        for p_idx, (px, py, pz, pw, ph, pd) in enumerate(free_prisms):
            # Early termination if we found a placement at the origin (0,0,0)
            # Since score is x+y+z, 0 is the minimum possible
            if best_score == 0:
                break
                
            for rw, rh, rd in rotations:
                if rw <= pw and rh <= ph and rd <= pd:
                    # The box fits in this prism
                    # Score: minimize distance from origin (0,0,0)
                    # Using sum of coordinates as a proxy for "compactness"
                    score = px + py + pz
                    
                    if score < best_score:
                        best_score = score
                        best_prism_idx = p_idx
                        best_rot = (rw, rh, rd)
                        best_pos = (px, py, pz)
                        
                        # If score is 0, it's the best possible, we can stop checking rotations for this prism
                        # But we still need to check other prisms? No, if score is 0, no other prism can be better.
                        # So we can break out of rotation loop, and the outer loop check will break.
                        if score == 0:
                            break
        
        if best_prism_idx != -1:
            # Place the box
            bw, bh, bd = best_rot
            bx, by, bz = best_pos
            
            # Record placement: (bin, x, y, z, w, h, d)
            placements[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Update free prisms
            # Remove the used prism
            used_prism = free_prisms.pop(best_prism_idx)
            ux, uy, uz, uw, uh, ud = used_prism
            
            # Generate new free prisms from the remainder
            # 1. Slice along X (Right)
            if bw < uw:
                free_prisms.append((ux + bw, uy, uz, uw - bw, uh, ud))
            
            # 2. Slice along Y (Top)
            # Note: This slice is only valid for the part of the prism that was "above" the box in the X-range [ux, ux+bw]
            # Our decomposition logic:
            # P1: [ux+bw, ux+uw] x [uy, uy+uh] x [uz, uz+ud]
            # P2: [ux, ux+bw] x [uy+bh, uy+uh] x [uz, uz+ud]
            # P3: [ux, ux+bw] x [uy, uy+bh] x [uz+bd, uz+ud]
            
            if bh < uh:
                free_prisms.append((ux, uy + bh, uz, bw, uh - bh, ud))
            
            # 3. Slice along Z (Back)
            if bd < ud:
                free_prisms.append((ux, uy, uz + bd, bw, bh, ud - bd))
                
        else:
            # Box cannot be placed
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
