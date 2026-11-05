"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: aim to finish well within 120s
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Generate all 6 unique rotations for a box (w, h, d)
    def get_rotations(w, h, d):
        rots = set()
        for perm in permutations((w, h, d)):
            rots.add(perm)
        return list(rots)
    
    # Pre-compute rotations for each box
    all_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending to place large boxes first
    # This is a heuristic: larger boxes are harder to place, so place them first
    box_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # We'll use a simple occupancy grid approach with a resolution
    # Choose resolution based on container size to keep grid manageable
    # For 60 boxes, we want a grid that's not too large
    # Let's use a resolution of about 5-10 cm or so, depending on container size
    # Actually, let's use a finer approach: track placed boxes and check overlap directly
    # but use a spatial hash or just brute force with early termination
    
    # Given the time constraint and 60 boxes, let's use a hybrid approach:
    # 1. Sort by volume descending
    # 2. For each box, try all rotations and find a valid position
    # 3. Use a simple heuristic for position: try to pack tightly
    
    # Let's use a 3D occupancy grid with reasonable resolution
    # Estimate resolution: we want the grid to be at most ~50x50x50 = 125k cells
    # But we need to be careful about time
    # Let's use a resolution that adapts to the container size
    
    # For simplicity and speed, let's use a direct overlap check with placed boxes
    # and a heuristic for candidate positions
    
    # Candidate positions: we'll try positions near existing boxes and corners
    # This is a common heuristic for 3D packing
    
    # First, let's try a simpler approach: use the seed algorithm but with rotations
    # and better ordering
    
    # Actually, let's implement a proper 3D packing algorithm with rotations
    # Using a grid-based approach for speed
    
    # Choose grid resolution
    # We want to be able to check if a box fits in a cell quickly
    # Let's use a resolution such that the grid is manageable
    # For a container of size W x H x D, use a resolution of about max(W,H,D)/20
    # But at least 1 unit resolution
    
    max_dim = max(W, H, D)
    if max_dim > 1000:
        res = max_dim / 20.0
    elif max_dim > 100:
        res = max_dim / 10.0
    else:
        res = 1.0
    
    # Grid dimensions
    gw = int(np.ceil(W / res)) + 1
    gh = int(np.ceil(H / res)) + 1
    gd = int(np.ceil(D / res)) + 1
    
    # If grid is too large, reduce resolution
    if gw * gh * gd > 200000:
        res *= 2
        gw = int(np.ceil(W / res)) + 1
        gh = int(np.ceil(H / res)) + 1
        gd = int(np.ceil(D / res)) + 1
    
    if gw * gh * gd > 500000:
        # Fall back to direct overlap checking
        return _load_direct(boxes, container, box_order, all_rots, start_time, deadline)
    
    # Initialize occupancy grid
    grid = np.zeros((gw, gh, gd), dtype=np.int8)
    
    # Placements
    placements = [None] * n
    
    # For each box in sorted order, try to place it
    for idx in box_order:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        placed = False
        
        for rot in all_rots[idx]:
            rw, rh, rd = rot
            # Convert to grid coordinates
            gw_box = int(np.ceil(rw / res))
            gh_box = int(np.ceil(rh / res))
            gd_box = int(np.ceil(rd / res))
            
            if gw_box > gw or gh_box > gh or gd_box > gd:
                continue
            
            # Find a valid position in the grid
            # Try positions from (0,0,0) upwards
            found = False
            for gz in range(0, gd - gd_box + 1):
                if found:
                    break
                for gy in range(0, gh - gh_box + 1):
                    if found:
                        break
                    for gx in range(0, gw - gw_box + 1):
                        # Check if this position is free
                        if np.any(grid[gx:gx+gw_box, gy:gy+gh_box, gz:gz+gd_box] > 0):
                            continue
                        # Place the box
                        grid[gx:gx+gw_box, gy:gy+gh_box, gz:gz+gd_box] = 1
                        # Convert back to real coordinates
                        px = gx * res
                        py = gy * res
                        pz = gz * res
                        placements[idx] = (0, px, py, pz, rw, rh, rd)
                        placed = True
                        found = True
                        break
            
            if placed:
                break
        
        if not placed:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced boxes
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements


def _load_direct(boxes, container, box_order, all_rots, start_time, deadline):
    """Direct overlap checking fallback."""
    W, H, D = container
    n = len(boxes)
    placements = [None] * n
    placed_boxes = []  # List of (x, y, z, w, h, d)
    
    for idx in box_order:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        placed = False
        
        for rot in all_rots[idx]:
            rw, rh, rd = rot
            if rw > W or rh > H or rd > D:
                continue
            
            # Try candidate positions
            # 1. Origin
            # 2. Near existing boxes
            candidates = [(0, 0, 0)]
            
            # Add positions near existing boxes
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Try to place adjacent to existing boxes
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))
            
            # Try each candidate position
            for (cx, cy, cz) in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check overlap with all placed boxes
                overlap = False
                for (px, py, pz, pw, ph, pd) in placed_boxes:
                    if (cx < px + pw and cx + rw > px and
                        cy < py + ph and cy + rh > py and
                        cz < pz + pd and cz + rd > pz):
                        overlap = True
                        break
                
                if not overlap:
                    placed_boxes.append((cx, cy, cz, rw, rh, rd))
                    placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                    placed = True
                    break
            
            if placed:
                break
        
        if not placed:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced boxes
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
