"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    deadline = time.time() + 105.0  # safety margin
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Precompute orientations for all boxes
    box_orientations = [get_orientations(w, h, d) for w, h, d in boxes]
    
    def shelf_pack_3d(ordered_indices):
        """
        3D shelf packing:
        - Y-axis is the stacking axis (shelves at different y levels)
        - Each shelf has a fixed height (determined by first box placed on it)
        - Within a shelf, boxes are placed in the X-Z plane
        - We use a simple "next fit" in X, and when X is full, we move to a new Z row within the same shelf
        - Actually, let's use a more standard 3D shelf approach:
          * Shelves are horizontal layers (fixed y, height h_shelf)
          * Within each shelf, we pack boxes in the X-Z plane using a simple grid or next-fit
          * For simplicity, we'll use: for each shelf, maintain a list of "rows" in Z
          * Each row has a fixed z_start and z_end (or just z_start and we track next_x)
          * Actually, let's simplify: for each shelf, we place boxes one by one
          * We try to place at the lowest available (x, z) position that fits
          * To keep it simple and fast, we'll use a "next fit decreasing" in X for each Z-row
          * But managing Z-rows is complex. Let's use a simpler approach:
          * For each shelf, we maintain a 2D grid of "next_x" for each z-position
          * But z is continuous... Let's use a list of "free rectangles" in X-Z for each shelf
          * This is getting complex. Let's use a simpler heuristic:
          * For each shelf, we place boxes in order of decreasing volume
          * We try to place each box at the smallest (x, z) position that fits
          * We maintain a list of placed boxes in the shelf and check for overlap
          * This is O(n^2) per shelf but n is small (60 boxes total)
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        
        # Shelves: list of dicts {'y': float, 'h': float, 'boxes': [(x, z, ow, od, oh)]}
        shelves = []
        current_y = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            # Try to place on existing shelves
            placed = False
            for shelf in shelves:
                # Try each orientation
                for ow, oh, od in orients:
                    if oh > shelf['h']:
                        continue
                    # Try to find a position (x, z) in this shelf where the box fits
                    # Check against all placed boxes in this shelf
                    found_pos = False
                    for x in range(0, W - ow + 1, max(1, W // 20)):  # Sample x positions
                        for z in range(0, D - od + 1, max(1, D // 20)):  # Sample z positions
                            # Check if this position is free
                            free = True
                            for px, pz, pow_, pod, poh in shelf['boxes']:
                                # Check overlap in X and Z
                                if not (x + ow <= px or px + pow_ <= x):
                                    if not (z + od <= pz or pz + pod <= z):
                                        free = False
                                        break
                            if free:
                                # Place the box here
                                placements[idx] = (0, x, int(shelf['y']), z, ow, oh, od)
                                shelf['boxes'].append((x, z, ow, od, oh))
                                total_volume += ow * oh * od
                                placed = True
                                found_pos = True
                                break
                        if found_pos:
                            break
                    if placed:
                        break
                if placed:
                    break
            
            if placed:
                continue
            
            # If not placed on existing shelves, try to start a new shelf
            best_orient = None
            for ow, oh, od in orients:
                if current_y + oh <= H and ow <= W and od <= D:
                    # Prefer orientations that minimize height to fit more shelves
                    if best_orient is None or oh < best_orient[1]:
                        best_orient = (ow, oh, od)
            
            if best_orient:
                ow, oh, od = best_orient
                new_shelf = {'y': current_y, 'h': oh, 'boxes': [(0, 0, ow, od, oh)]}
                shelves.append(new_shelf)
                current_y += oh
                placements[idx] = (0, 0, int(current_y - oh), 0, ow, oh, od)
                total_volume += ow * oh * od
            # Else: box cannot fit in remaining height, leave it out
        
        return placements, total_volume
    
    # Generate different sorting strategies
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # Sum of dims
        lambda i: boxes[i][1],  # Height
        lambda i: boxes[i][0],  # Width
        lambda i: boxes[i][2],  # Depth
        lambda i: -min(boxes[i][0], boxes[i][1], boxes[i][2]),  # Min dim (ascending)
        lambda i: boxes[i][0] * boxes[i][1],  # Base area
        lambda i: boxes[i][1] * boxes[i][2],  # Side area
        lambda i: boxes[i][0] * boxes[i][2],  # Top area
    ]
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    for sort_key in strategies:
        if time.time() > deadline - 15:
            break
        
        ordered = sorted(indexed_boxes, key=sort_key, reverse=True)
        p, v = shelf_pack_3d(ordered)
        if v > best_volume:
            best_placements = p
            best_volume = v
    
    return best_placements
# EVOLVE-BLOCK-END
