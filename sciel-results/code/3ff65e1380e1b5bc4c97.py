"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import product

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Initial placement list
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n

    def can_place(x, y, z, w, h, d, placed_list):
        """Check if box at (x,y,z) with dims (w,h,d) doesn't overlap with any placed box."""
        for px, py, pz, pw, ph, pd in placed_list:
            # Overlap check
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def try_all_orientations(box_dims):
        """For a single box, generate all 6 unique orientations."""
        w, h, d = box_dims
        orientations = []
        seen = set()
        for perm in set(product((w, h, d), repeat=3)):
            if sorted(perm) == sorted((w, h, d)):
                if perm not in seen:
                    seen.add(perm)
                    orientations.append(perm)
        return orientations

    def get_candidate_positions(placed, ow, oh, od, W, H, D):
        """Generate candidate positions for a box with dimensions (ow, oh, od)."""
        positions = set()
        positions.add((0, 0, 0))
        
        for px, py, pz, pw, ph, pd in placed:
            # Right of box
            if px + pw + ow <= W:
                positions.add((px + pw, py, pz))
                positions.add((px + pw, py + ph, pz))
                positions.add((px + pw, py, pz + pd))
                positions.add((px + pw, py + ph, pz + pd))
            # Above box (in y)
            if py + ph + oh <= H:
                positions.add((px, py + ph, pz))
                positions.add((px + pw, py + ph, pz))
                positions.add((px, py + ph, pz + pd))
                positions.add((px + pw, py + ph, pz + pd))
            # In front of box (in z)
            if pz + pd + od <= D:
                positions.add((px, py, pz + pd))
                positions.add((px + pw, py, pz + pd))
                positions.add((px, py + ph, pz + pd))
                positions.add((px + pw, py + ph, pz + pd))
        
        return positions

    def greedy_fill(boxes_sorted, container, best_fit=True, strategy='min_coord'):
        """Greedy first-fit decreasing with multiple orientation tries."""
        W, H, D = container
        placed = []  # list of (x, y, z, w, h, d)
        result = {}  # box_idx -> (x, y, z, w, h, d)
        
        for idx in boxes_sorted:
            w, h, d = boxes[idx]
            if time.time() - start_time > DEADLINE:
                break
            
            orientations = try_all_orientations((w, h, d))
            best_orient = None
            best_pos = None
            best_score = float('inf') if best_fit else -1
            
            # Try each orientation
            for ow, oh, od in orientations:
                # Check if fits in container
                if ow > W or oh > H or od > D:
                    continue
                
                # Try to find a position
                positions_to_try = get_candidate_positions(placed, ow, oh, od, W, H, D)
                
                # Try each position
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    if can_place(tx, ty, tz, ow, oh, od, placed):
                        # Score: prefer positions closer to origin (less wasted space)
                        if strategy == 'min_coord':
                            score = tx + ty + tz
                        elif strategy == 'min_max':
                            score = max(tx, ty, tz)
                        elif strategy == 'min_volume':
                            score = tx * ty * tz + tx + ty + tz
                        else:
                            score = tx + ty + tz
                        
                        if best_fit:
                            if score < best_score:
                                best_score = score
                                best_orient = (ow, oh, od)
                                best_pos = (tx, ty, tz)
                        else:
                            if score > best_score:
                                best_score = score
                                best_orient = (ow, oh, od)
                                best_pos = (tx, ty, tz)
            
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
                result[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
        
        return result, placed

    def compute_fill(result, container):
        """Compute total volume of placed boxes."""
        total = 0
        for idx, (x, y, z, w, h, d) in result.items():
            total += w * h * d
        return total / container_vol

    # Strategy 1: Sort by volume descending
    def sort_by_volume(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Strategy 2: Sort by max dimension descending
    def sort_by_max_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    # Strategy 3: Sort by volume/shortest_dim (prefer cubical)
    def sort_by_cubicality(boxes):
        return sorted(range(len(boxes)), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]) / min(boxes[i]) if min(boxes[i]) > 0 else 0, reverse=True)
    
    # Strategy 4: Sort by elongation (prefer less elongated)
    def sort_by_elongation(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]) / min(boxes[i]) if min(boxes[i]) > 0 else float('inf'), reverse=False)

    # Strategy 5: Sort by surface area descending
    def sort_by_surface(boxes):
        return sorted(range(len(boxes)), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)

    best_result = {}
    best_fill = 0.0

    strategies = [
        (sort_by_volume, True, 'min_coord'),
        (sort_by_volume, False, 'min_coord'),
        (sort_by_max_dim, True, 'min_coord'),
        (sort_by_max_dim, False, 'min_coord'),
        (sort_by_cubicality, True, 'min_coord'),
        (sort_by_elongation, True, 'min_coord'),
        (sort_by_surface, True, 'min_coord'),
        (sort_by_volume, True, 'min_max'),
        (sort_by_volume, False, 'min_max'),
        (sort_by_max_dim, True, 'min_max'),
        (sort_by_cubicality, True, 'min_max'),
    ]

    for strat, best_fit, strategy in strategies:
        if time.time() - start_time > DEADLINE:
            break
        
        boxes_sorted = strat(boxes)
        result, placed = greedy_fill(boxes_sorted, container, best_fit, strategy)
        
        fill = compute_fill(result, container)
        if fill > best_fill:
            best_fill = fill
            best_result = result
    
    # Try a second pass: for boxes not placed, try again with remaining space
    # This is a simple improvement - try to fit remaining boxes using current placed set
    if best_result and time.time() - start_time < DEADLINE - 5:
        placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in best_result.values()]
        
        # Get unplaced boxes
        unplaced = [i for i in range(n) if i not in best_result]
        
        # Try to place unplaced boxes one by one
        for idx in unplaced:
            if time.time() - start_time > DEADLINE:
                break
            w, h, d = boxes[idx]
            orientations = try_all_orientations((w, h, d))
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                positions_to_try = get_candidate_positions(placed, ow, oh, od, W, H, D)
                
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    if can_place(tx, ty, tz, ow, oh, od, placed):
                        best_result[idx] = (tx, ty, tz, ow, oh, od)
                        placed.append((tx, ty, tz, ow, oh, od))
                        break
                if idx in best_result:
                    break
    
    # Local search: try to improve by moving boxes to better positions
    if best_result and time.time() - start_time < DEADLINE - 10:
        improved = True
        max_iterations = 5
        iteration = 0
        while improved and iteration < max_iterations and time.time() - start_time < DEADLINE - 5:
            improved = False
            iteration += 1
            placed_list = list(best_result.items())
            
            for i, (idx1, pos1) in enumerate(placed_list):
                if time.time() - start_time > DEADLINE:
                    break
                x1, y1, z1, w1, h1, d1 = pos1
                
                # Try to move this box to a better position
                orientations1 = try_all_orientations((w1, h1, d1))
                other_placed = [(p[0], p[1], p[2], p[3], p[4], p[5]) for j, (idx2, p) in enumerate(placed_list) if j != i]
                
                best_new_pos = None
                best_new_orient = None
                best_score = x1 + y1 + z1
                
                for ow, oh, od in orientations1:
                    if ow > W or oh > H or od > D:
                        continue
                    
                    positions_to_try = get_candidate_positions(other_placed, ow, oh, od, W, H, D)
                    
                    for (tx, ty, tz) in positions_to_try:
                        if tx + ow > W or ty + oh > H or tz + od > D:
                            continue
                        if can_place(tx, ty, tz, ow, oh, od, other_placed):
                            score = tx + ty + tz
                            if score < best_score:
                                best_score = score
                                best_new_pos = (tx, ty, tz)
                                best_new_orient = (ow, oh, od)
                
                if best_new_pos is not None and (best_new_pos != (x1, y1, z1) or best_new_orient != (w1, h1, d1)):
                    best_result[idx1] = best_new_pos + best_new_orient
                    improved = True
    
    # Build final placements list
    for i in range(n):
        if i in best_result:
            x, y, z, w, h, d = best_result[i]
            placements[i] = (0, x, y, z, w, h, d)
        else:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
