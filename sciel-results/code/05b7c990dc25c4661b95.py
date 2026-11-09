"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import permutations
from collections import defaultdict

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Initial placement list
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n

    def get_orientations(w, h, d):
        """Get all unique orientations of a box."""
        perms = set(permutations((w, h, d)))
        return list(perms)

    def can_place(x, y, z, w, h, d, placed_list):
        """Check if box at (x,y,z) with dims (w,h,d) doesn't overlap with any placed box."""
        for px, py, pz, pw, ph, pd in placed_list:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def get_candidate_positions(placed, ow, oh, od):
        """Generate candidate positions: origin + all face-adjacent positions + some interior points."""
        positions = set()
        positions.add((0, 0, 0))
        
        # For each placed box, generate positions adjacent to its faces
        for px, py, pz, pw, ph, pd in placed:
            # x positions: at left face, right face, and slightly offset
            x_opts = set()
            x_opts.add(px)
            x_opts.add(px + pw)
            # y positions: at bottom face, top face
            y_opts = set()
            y_opts.add(py)
            y_opts.add(py + ph)
            # z positions: at back face, front face
            z_opts = set()
            z_opts.add(pz)
            z_opts.add(pz + pd)
            
            for cx in x_opts:
                for cy in y_opts:
                    for cz in z_opts:
                        if 0 <= cx <= W - ow and 0 <= cy <= H - oh and 0 <= cz <= D - od:
                            positions.add((cx, cy, cz))
        
        # Also try positions at max extents (pushed to far walls)
        if placed:
            max_x = min(W - ow, max(px + pw for px, py, pz, pw, ph, pd in placed) if placed else 0)
            max_y = min(H - oh, max(py + ph for px, py, pz, pw, ph, pd in placed) if placed else 0)
            max_z = min(D - od, max(pz + pd for px, py, pz, pw, ph, pd in placed) if placed else 0)
            positions.add((max_x, 0, 0))
            positions.add((0, max_y, 0))
            positions.add((0, 0, max_z))
            positions.add((max_x, max_y, 0))
            positions.add((max_x, 0, max_z))
            positions.add((0, max_y, max_z))
            positions.add((max_x, max_y, max_z))
        
        return list(positions)

    def score_position(x, y, z, ow, oh, od, placed):
        """Score a position: prefer tight packing (minimize gaps to existing boxes and walls)."""
        # Base score: prefer lower coordinates (tighter to origin)
        base = -(x + y + z)
        
        # Bonus for being adjacent to existing boxes (reduces fragmentation)
        adjacency_bonus = 0
        for px, py, pz, pw, ph, pd in placed:
            # Check if touching on any face
            if abs(x - (px + pw)) < 1e-9 or abs(px - (x + ow)) < 1e-9:
                adjacency_bonus += 1
            if abs(y - (py + ph)) < 1e-9 or abs(py - (y + oh)) < 1e-9:
                adjacency_bonus += 1
            if abs(z - (pz + pd)) < 1e-9 or abs(pz - (z + od)) < 1e-9:
                adjacency_bonus += 1
        
        return base - adjacency_bonus * 0.1  # Slight preference for adjacency

    def greedy_fill(boxes_sorted, container, placed_initial=None):
        """Greedy first-fit with multiple orientation tries and scoring."""
        W, H, D = container
        placed = list(placed_initial) if placed_initial else []
        result = {}
        
        for idx in boxes_sorted:
            w, h, d = boxes[idx]
            if time.time() - start_time > DEADLINE:
                break
            
            orientations = get_orientations(w, h, d)
            best_orient = None
            best_pos = None
            best_score = float('inf')
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                positions_to_try = get_candidate_positions(placed, ow, oh, od)
                
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    if can_place(tx, ty, tz, ow, oh, od, placed):
                        sc = score_position(tx, ty, tz, ow, oh, od, placed)
                        if sc < best_score:
                            best_score = sc
                            best_orient = (ow, oh, od)
                            best_pos = (tx, ty, tz)
            
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
                result[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
        
        return result, placed

    def compute_fill(result):
        """Compute total volume of placed boxes."""
        total = 0
        for idx, (x, y, z, w, h, d) in result.items():
            total += w * h * d
        return total / container_vol

    def local_search_improve(result, container, max_iter=200):
        """Try to improve placement by moving boxes to better positions."""
        if not result:
            return result, 0.0
        
        W, H, D = container
        current_fill = compute_fill(result)
        best_result = dict(result)
        best_fill = current_fill
        
        placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in result.values()]
        
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            # Try moving each placed box to a better position
            for idx in list(result.keys()):
                if time.time() - start_time > DEADLINE:
                    break
                
                ox, oy, oz, ow, oh, od = result[idx]
                # Remove from placed
                placed_copy = [p for p in placed if p != (ox, oy, oz, ow, oh, od)]
                
                orientations = get_orientations(ow, oh, od)
                best_new_pos = None
                best_new_orient = None
                best_score = score_position(ox, oy, oz, ow, oh, od, placed_copy)
                
                for nwo, nho, nod in orientations:
                    if nwo > W or nho > H or nod > D:
                        continue
                    
                    positions_to_try = get_candidate_positions(placed_copy, nwo, nho, nod)
                    
                    for (tx, ty, tz) in positions_to_try:
                        if tx + nwo > W or ty + nho > H or tz + nod > D:
                            continue
                        if can_place(tx, ty, tz, nwo, nho, nod, placed_copy):
                            sc = score_position(tx, ty, tz, nwo, nho, nod, placed_copy)
                            if sc < best_score - 1e-9:
                                best_score = sc
                                best_new_pos = (tx, ty, tz)
                                best_new_orient = (nwo, nho, nod)
                
                if best_new_pos is not None:
                    # Apply the move
                    best_result[idx] = (best_new_pos[0], best_new_pos[1], best_new_pos[2], best_new_orient[0], best_new_orient[1], best_new_orient[2])
                    placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in best_result.values()]
                    new_fill = compute_fill(best_result)
                    if new_fill > best_fill:
                        best_fill = new_fill
                        improved = True
                    else:
                        # Revert
                        best_result[idx] = (ox, oy, oz, ow, oh, od)
                        placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in best_result.values()]
            
            if not improved:
                break
        
        return best_result, best_fill

    # Sorting strategies
    def sort_by_volume(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_surface_area(boxes):
        return sorted(range(len(boxes)), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    def sort_by_volume_to_max_ratio(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] / max(boxes[i]) if max(boxes[i]) > 0 else 0, reverse=True)
    
    def sort_by_min_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)

    best_result = {}
    best_fill = 0.0

    strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_surface_area,
        sort_by_volume_to_max_ratio,
        sort_by_min_dim,
    ]

    for strat in strategies:
        if time.time() - start_time > DEADLINE:
            break
        
        boxes_sorted = strat(boxes)
        result, placed = greedy_fill(boxes_sorted, container)
        
        fill = compute_fill(result)
        if fill > best_fill:
            best_fill = fill
            best_result = dict(result)
    
    # Local search improvement on the best result
    if best_result and time.time() - start_time < DEADLINE - 10:
        best_result, best_fill = local_search_improve(best_result, container)
    
    # Second pass: try to place remaining boxes
    if best_result and time.time() - start_time < DEADLINE - 5:
        placed = [(x, y, z, w, h, d) for (x, y, z, w, h, d) in best_result.values()]
        unplaced = [i for i in range(n) if i not in best_result]
        
        # Sort unplaced by volume descending
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in unplaced:
            if time.time() - start_time > DEADLINE:
                break
            w, h, d = boxes[idx]
            orientations = get_orientations(w, h, d)
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                positions_to_try = get_candidate_positions(placed, ow, oh, od)
                
                for (tx, ty, tz) in positions_to_try:
                    if tx + ow > W or ty + oh > H or tz + od > D:
                        continue
                    if can_place(tx, ty, tz, ow, oh, od, placed):
                        best_result[idx] = (tx, ty, tz, ow, oh, od)
                        placed.append((tx, ty, tz, ow, oh, od))
                        break
                if idx in best_result:
                    break
    
    # Final local search if time permits
    if best_result and time.time() - start_time < DEADLINE - 3:
        best_result, best_fill = local_search_improve(best_result, container, max_iter=50)
    
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
