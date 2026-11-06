"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Get unique dimensions for grid resolution
    all_dims = set()
    for w, h, d in boxes:
        all_dims.add(w)
        all_dims.add(h)
        all_dims.add(d)
    all_dims.add(W)
    all_dims.add(H)
    all_dims.add(D)
    
    # Use a reasonable grid resolution
    # We'll use a grid where each cell is 1 unit if dimensions are integers,
    # otherwise we'll scale
    # For simplicity and speed, use a coarse grid
    grid_res = 1  # 1 unit per cell
    
    # Create 3D occupancy grid
    grid_w = W // grid_res + 1
    grid_h = H // grid_res + 1
    grid_d = D // grid_res + 1
    
    # Use a 3D numpy array for occupancy (0 = empty, 1 = occupied)
    occ = np.zeros((grid_w, grid_h, grid_d), dtype=np.uint8)
    
    # Precompute all 6 orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        orients = set()
        for perm in permutations([w, h, d]):
            orients.add(perm)
        box_orientations.append(list(orients))
    
    # Sort boxes by volume descending (larger first)
    box_indices = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    placements = [None] * n
    
    def can_place(ow, oh, od, ox, oy, oz):
        """Check if box with dims (ow,oh,od) can be placed at (ox,oy,oz)"""
        x1, x2 = ox // grid_res, (ox + ow - 1) // grid_res
        y1, y2 = oy // grid_res, (oy + oh - 1) // grid_res
        z1, z2 = oz // grid_res, (oz + od - 1) // grid_res
        
        if x2 >= grid_w or y2 >= grid_h or z2 >= grid_d:
            return False
        
        # Check if any cell in the region is occupied
        if np.any(occ[x1:x2+1, y1:y2+1, z1:z2+1] > 0):
            return False
        return True
    
    def place_box(ow, oh, od, ox, oy, oz):
        """Mark box as occupied in grid"""
        x1, x2 = ox // grid_res, (ox + ow - 1) // grid_res
        y1, y2 = oy // grid_res, (oy + oh - 1) // grid_res
        z1, z2 = oz // grid_res, (oz + od - 1) // grid_res
        occ[x1:x2+1, y1:y2+1, z1:z2+1] = 1
    
    def find_best_position(ow, oh, od):
        """Find the best (lowest z, then lowest y, then lowest x) position for a box"""
        # Try all possible positions, prioritizing lowest z, then y, then x
        best_pos = None
        best_score = float('inf')
        
        for oz in range(0, D - od + 1, grid_res):
            for oy in range(0, H - oh + 1, grid_res):
                for ox in range(0, W - ow + 1, grid_res):
                    if can_place(ow, oh, od, ox, oy, oz):
                        score = oz * 10000 + oy * 100 + ox
                        if score < best_score:
                            best_score = score
                            best_pos = (ox, oy, oz)
                        break  # Found lowest x for this y, z
                else:
                    continue
                break  # Found lowest y for this z
            else:
                continue
            break  # Found lowest z
        
        return best_pos
    
    # Try to place each box
    placed_count = 0
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        placed = False
        
        for ow, oh, od in box_orientations[idx]:
            # Check if this orientation fits in container
            if ow > W or oh > H or od > D:
                continue
            
            pos = find_best_position(ow, oh, od)
            if pos is not None:
                ox, oy, oz = pos
                place_box(ow, oh, od, ox, oy, oz)
                placements[idx] = (0, ox, oy, oz, ow, oh, od)
                placed = True
                placed_count += 1
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
