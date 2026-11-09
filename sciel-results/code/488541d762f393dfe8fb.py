"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations

def _try_place(box_dims, container, placed):
    """Try to place a box with any axis rotation using max-min heuristic.
    Returns (x, y, z, w, h, d) if placed, else None."""
    W, H, D = container
    # Generate all 6 rotations
    rotations = set()
    for perm in permutations(box_dims):
        rotations.add(tuple(perm))
    rotations = list(rotations)
    
    best_pos = None
    best_score = -1
    
    for (w, h, d) in rotations:
        if w > W or h > H or d > D:
            continue
        # Use a grid-based approach to find a valid position
        # For efficiency, we check a coarse grid first, then refine
        # Simple approach: try to place at (0,0,0) then shift along axes
        # Better: use a priority queue or sorted list of potential positions
        
        # For simplicity and speed with 60 boxes, we use a heuristic:
        # Try positions based on existing boxes' edges
        candidate_xs = [0] + [x + pw for (_, px, py, pz, pw, ph, pd) in placed for pw_ in [pw] if x + pw <= W]
        # Actually, let's just do a simple scan for small containers or use a smarter method
        
        # Given the constraint of 120s for 60 boxes, let's use a simpler but effective method:
        # Sort rotations by volume (largest first) and try to place using a "skyline" or simple grid
        pass
    
    # Let's use a different approach: for each rotation, try to find the lowest, most forward, most left position
    # We'll use a 3D grid if container is small enough, otherwise use a simpler heuristic
    
    # For robustness, let's use a simple iterative placement:
    # For each rotation, try x from 0 to W-w, y from 0 to H-h, z from 0 to D-d
    # This is O(W*H*D) which could be slow. Let's optimize.
    
    # Better: for each rotation, collect candidate positions from existing boxes
    # A position is valid if it doesn't overlap with any placed box
    # Candidate positions: x in {0} U {x_i + w_i}, y in {0} U {y_i + h_i}, z in {0} U {z_i + d_i}
    
    for (w, h, d) in rotations:
        if w > W or h > H or d > D:
            continue
        
        cx = [0]
        cy = [0]
        cz = [0]
        for (_, px, py, pz, pw, ph, pd) in placed:
            nx = px + pw
            if nx <= W - w:
                cx.append(nx)
            ny = py + ph
            if ny <= H - h:
                cy.append(ny)
            nz = pz + pd
            if nz <= D - d:
                cz.append(nz)
        
        # Also add some intermediate positions? For now, just these
        # Find the position with minimum (z, y, x) that is valid
        best_local = None
        best_local_score = None
        
        for x in cx:
            if x + w > W:
                continue
            for y in cy:
                if y + h > H:
                    continue
                for z in cz:
                    if z + d > D:
                        continue
                    # Check overlap with all placed boxes
                    overlap = False
                    for (_, px, py, pz, pw, ph, pd) in placed:
                        if (x < px + pw and px < x + w and
                            y < py + ph and py < y + h and
                            z < pz + pd and pz < z + d):
                            overlap = True
                            break
                    if not overlap:
                        # Score: prefer lower z, then lower y, then lower x
                        score = (z, y, x)
                        if best_local_score is None or score < best_local_score:
                            best_local_score = score
                            best_local = (x, y, z, w, h, d)
        
        if best_local is not None:
            # Compare with global best
            # Prefer larger dimensions? Or just take the first valid?
            # Let's prefer the rotation that allows more space, i.e., the one with smallest max extent
            # For now, just take the first valid rotation found with best position
            if best_pos is None:
                best_pos = best_local
                best_score = best_local_score
            else:
                # Compare: prefer lower z, then lower y, then lower x
                if best_local_score < best_score:
                    best_pos = best_local
                    best_score = best_local_score
    
    return best_pos

def load(boxes, container):
    W, H, D = container
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    placed = []  # List of (bin, x, y, z, w, h, d)
    results = [None] * len(boxes)
    
    for i, (idx, w, h, d, vol) in indexed_boxes:
        pos = _try_place((w, h, d), container, placed)
        if pos is not None:
            x, y, z, pw, ph, pd = pos
            placed.append((0, x, y, z, pw, ph, pd))
            results[idx] = (0, x, y, z, pw, ph, pd)
        else:
            results[idx] = (-1, 0, 0, 0, w, h, d)
    
    return results
# EVOLVE-BLOCK-END
