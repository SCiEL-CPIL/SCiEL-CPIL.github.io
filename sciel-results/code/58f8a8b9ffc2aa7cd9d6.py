"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate all 6 rotations for each box
    def rotations(w, h, d):
        # Unique permutations of (w, h, d)
        return list(set(itertools.permutations((w, h, d))))
    
    # For each box, compute possible placements in a "skyline" or simple grid approach
    # We'll use a simple shelf-based approach but with rotation and sorting.
    
    # Sort boxes by volume descending to place larger boxes first
    box_indices = list(range(n))
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in box_indices]
    sorted_indices = sorted(box_indices, key=lambda i: volumes[i], reverse=True)
    
    # We'll use a simple 3D grid occupancy check, but that's O(W*H*D) which might be too slow.
    # Instead, we'll use a list of placed boxes and check for overlap.
    
    placed = []  # List of (x, y, z, w, h, d) for placed boxes
    
    def can_place(x, y, z, w, h, d):
        # Check if the box at (x,y,z) with dims (w,h,d) overlaps with any placed box
        for px, py, pz, pw, ph, pd in placed:
            # Check overlap in all three dimensions
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return False
        # Check if within container
        if x + w > W or y + h > H or z + d > D:
            return False
        return True
    
    # For efficiency, we'll use a simple heuristic: try to place each box at the lowest
    # possible position, scanning x, then y, then z.
    # To make this faster, we'll use a grid-based approach with a resolution of 1 unit
    # if the container dimensions are small enough, otherwise we'll use a coarser approach.
    
    # Let's use a hybrid approach: if the container is small (W*H*D < 1000000), use grid.
    # Otherwise, use the overlap check with a limited search.
    
    total_volume = W * H * D
    if total_volume <= 2000000:
        # Use a 3D grid for occupancy
        grid = np.zeros((W, H, D), dtype=bool)
        
        def can_place_grid(x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            if grid[x:x+w, y:y+h, z:z+d].any():
                return False
            return True
        
        def place_grid(x, y, z, w, h, d):
            grid[x:x+w, y:y+h, z:z+d] = True
        
        result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
        for i in sorted_indices:
            w, h, d = boxes[i]
            placed = False
            for rot_w, rot_h, rot_d in rotations(w, h, d):
                if placed:
                    break
                # Try to find a position
                # Scan z (bottom to top), then y, then x
                for z in range(D - rot_d + 1):
                    if placed:
                        break
                    for y in range(H - rot_h + 1):
                        if placed:
                            break
                        for x in range(W - rot_w + 1):
                            if can_place_grid(x, y, z, rot_w, rot_h, rot_d):
                                place_grid(x, y, z, rot_w, rot_h, rot_d)
                                result[i] = (0, x, y, z, rot_w, rot_h, rot_d)
                                placed = True
                                break
            if not placed:
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result
    else:
        # Use overlap check with a limited search strategy
        # We'll use a shelf-based approach with rotation
        result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
        # Simple shelf packing: maintain shelves in the z-direction
        # Each shelf has a height, and we pack boxes along x and y within the shelf
        
        # For simplicity, we'll use a row-based approach within each shelf
        shelves = []  # List of (z_start, z_end, shelves_data)
        
        for i in sorted_indices:
            w, h, d = boxes[i]
            placed = False
            for rot_w, rot_h, rot_d in rotations(w, h, d):
                if placed:
                    break
                # Try to place in existing shelves
                for shelf in shelves:
                    z_start, z_end, rows = shelf
                    shelf_h = z_end - z_start
                    if rot_d > shelf_h:
                        continue
                    # Try to place in existing rows
                    for row in rows:
                        y_start, y_end, x_next = row
                        row_h = y_end - y_start
                        if rot_h > row_h:
                            continue
                        if x_next + rot_w <= W:
                            # Place here
                            x = x_next
                            y = y_start
                            z = z_start
                            placed_box = (x, y, z, rot_w, rot_h, rot_d)
                            placed.append(placed_box)
                            row[2] = x + rot_w  # Update x_next
                            result[i] = (0, x, y, z, rot_w, rot_h, rot_d)
                            placed = True
                            break
                    if placed:
                        break
                    # Try to add a new row in this shelf
                    if z_start + sum(r[1] - r[0] for r in rows) + rot_h <= z_end:
                        y_start = z_start + sum(r[1] - r[0] for r in rows)
                        y_end = y_start + rot_h
                        new_row = [y_start, y_end, rot_w]
                        rows.append(new_row)
                        x = 0
                        y = y_start
                        z = z_start
                        placed_box = (x, y, z, rot_w, rot_h, rot_d)
                        placed.append(placed_box)
                        result[i] = (0, x, y, z, rot_w, rot_h, rot_d)
                        placed = True
                        break
                if placed:
                    break
                # Try to create a new shelf
                current_max_z = max((s[1] for s in shelves), default=0)
                if current_max_z + rot_d <= D:
                    z_start = current_max_z
                    z_end = z_start + rot_d
                    new_row = [0, rot_h, rot_w]
                    new_shelf = (z_start, z_end, [new_row])
                    shelves.append(new_shelf)
                    x = 0
                    y = 0
                    z = z_start
                    placed_box = (x, y, z, rot_w, rot_h, rot_d)
                    placed.append(placed_box)
                    result[i] = (0, x, y, z, rot_w, rot_h, rot_d)
                    placed = True
                    break
            if not placed:
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result

# EVOLVE-BLOCK-END
