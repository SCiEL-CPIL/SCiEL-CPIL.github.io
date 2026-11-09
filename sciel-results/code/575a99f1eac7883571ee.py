"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import product
import random

def load(boxes, container):
    start_time = time.time()
    # Safety margin: 150s limit, use 140s
    DEADLINE = start_time + 140.0 

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Initial placement list
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n

    def get_rotations(w, h, d):
        """Return unique rotations of a box (w, h, d)."""
        seen = set()
        rots = []
        for perm in product((w, h, d), repeat=3):
            if sorted(perm) == sorted((w, h, d)):
                if perm not in seen:
                    seen.add(perm)
                    rots.append(perm)
        return rots

    def can_place(x, y, z, w, h, d, placed_list):
        """Check if box at (x,y,z) with dims (w,h,d) doesn't overlap with any placed box."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed_list:
            # Overlap check
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def get_grid_candidates(placed, ow, oh, od):
        """
        Generate candidate positions based on the 'extreme points' grid.
        Coordinates are 0 or the 'end' of any placed box along that axis.
        """
        if not placed:
            return [(0, 0, 0)]
        
        xs = {0}
        ys = {0}
        zs = {0}
        
        for px, py, pz, pw, ph, pd in placed:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)
                
        # Filter coordinates that allow the box to fit
        xs = sorted([x for x in xs if x + ow <= W])
        ys = sorted([y for y in ys if y + oh <= H])
        zs = sorted([z for z in zs if z + od <= D])
        
        if not xs or not ys or not zs:
            return []
            
        # Generate combinations
        # To keep it fast, we limit the number of candidates or rely on early termination in the caller
        # However, for 60 boxes, |xs|, |ys|, |zs| can be up to 60. 60^3 = 216,000.
        # This is too many to check all for 'Best Fit'.
        # We will return them sorted by "compactness" (x+y+z) so First Fit works well.
        candidates = []
        for x in xs:
            for y in ys:
                for z in zs:
                    candidates.append((x, y, z))
                    
        # Sort by sum of coordinates to prefer "bottom-left-front" packing
        candidates.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
        
        return candidates

    def greedy_fill(boxes_sorted, container, use_best_fit=False):
        """
        Greedy fill using the grid candidate generation.
        If use_best_fit is True, it checks all candidates and picks the best (min x+y+z).
        If False, it picks the first valid candidate (First Fit).
        """
        W, H, D = container
        placed = []  # list of (x, y, z, w, h, d)
        result = {}  # box_idx -> (x, y, z, w, h, d)
        
        for idx in boxes_sorted:
            if time.time() > DEADLINE:
                break
                
            w, h, d = boxes[idx]
            rotations = get_rotations(w, h, d)
            
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            for ow, oh, od in rotations:
                if ow > W or oh > H or od > D:
                    continue
                
                candidates = get_grid_candidates(placed, ow, oh, od)
                
                if not candidates:
                    continue
                
                if use_best_fit:
                    # Check all candidates and find the one with min x+y+z
                    for (tx, ty, tz) in candidates:
                        if can_place(tx, ty, tz, ow, oh, od, placed):
                            score = tx + ty + tz
                            if score < best_score:
                                best_score = score
                                best_pos = (tx, ty, tz)
                                best_orient = (ow, oh, od)
                else:
                    # First Fit: candidates are already sorted by compactness
                    for (tx, ty, tz) in candidates:
                        if can_place(tx, ty, tz, ow, oh, od, placed):
                            best_pos = (tx, ty, tz)
                            best_orient = (ow, oh, od)
                            break # Stop at first valid for this orientation
                    
                    # If we found a spot in this orientation, do we check other orientations?
                    # Usually, First Fit Decreasing implies we just take the first valid spot overall.
                    # But checking other orientations might yield a "better" fit (less waste).
                    # For speed, let's stick to the first valid spot found across orientations 
                    # if we are in First Fit mode, OR we can break out of the rotation loop 
                    # once we find *any* valid spot, assuming the first rotation is the "canonical" one.
                    # However, rotating might allow fitting in a tight space.
                    # Let's try all rotations but break early if we find a spot with score 0 (corner)?
                    # No, let's just take the first valid spot found in the sorted candidate list 
                    # for the *first* valid rotation.
                    # Actually, standard FFD tries the item in its given orientation. 
                    # Here we have rotations. Let's try all rotations but keep the 
                    # "best" (min x+y+z) among the *first valid* spots? 
                    # That's expensive.
                    # Let's simplify: For First Fit mode, we iterate rotations. 
                    # For each rotation, we find the first valid candidate. 
                    # We keep the overall best (min x+y+z) among these first-valid spots.
                    if best_pos is not None:
                         score = best_pos[0] + best_pos[1] + best_pos[2]
                         # We don't update best_score here because we are in First Fit mode 
                         # and we want to compare across rotations?
                         # Let's just break if we find a spot at (0,0,0) or similar?
                         # No, let's collect the first valid spot for each rotation and pick the best.
                         # This is safe and not too slow.
                         if best_score > score:
                             best_score = score
                             # best_pos and best_orient are already updated in the inner loop?
                             # Wait, the inner loop breaks on first valid. 
                             # So best_pos holds the first valid for this rotation.
                             # We need to compare this against previous rotations.
                             # The logic above is slightly flawed. Let's refine.
                             
                    # Refined First Fit Logic:
                    # We want the "best" first-fit spot across all rotations.
                    # "Best" defined by min(x+y+z).
                    # We already have best_pos/best_orient from previous rotations (or None).
                    # If we found a spot in this rotation (best_pos is not None from the loop above? 
                    # No, best_pos is global. We need local variables per rotation).
                    
                    # Let's restart the logic for clarity.
                    # This block is getting messy. Let's implement a helper or clear variables.
                    pass 

            # Re-implementing the loop logic cleanly below
            
        # The loop above was a sketch. Let's write the actual implementation.
        
        return result, placed

    # --- Corrected Implementation ---
    
    def solve_instance(boxes, container, order, use_best_fit=False):
        W, H, D = container
        placed = []
        result = {}
        
        for idx in order:
            if time.time() > DEADLINE:
                break
                
            w, h, d = boxes[idx]
            rotations = get_rotations(w, h, d)
            
            best_pos_global = None
            best_orient_global = None
            best_score_global = float('inf')
            
            for ow, oh, od in rotations:
                if ow > W or oh > H or od > D:
                    continue
                
                candidates = get_grid_candidates(placed, ow, oh, od)
                if not candidates:
                    continue
                
                if use_best_fit:
                    # Check all
                    for (tx, ty, tz) in candidates:
                        if can_place(tx, ty, tz, ow, oh, od, placed):
                            score = tx + ty + tz
                            if score < best_score_global:
                                best_score_global = score
                                best_pos_global = (tx, ty, tz)
                                best_orient_global = (ow, oh, od)
                else:
                    # First Fit: Find the first valid candidate in the sorted list
                    for (tx, ty, tz) in candidates:
                        if can_place(tx, ty, tz, ow, oh, od, placed):
                            score = tx + ty + tz
                            if score < best_score_global:
                                best_score_global = score
                                best_pos_global = (tx, ty, tz)
                                best_orient_global = (ow, oh, od)
                            break # Only consider the first valid spot for this orientation
                
            if best_pos_global is not None:
                placed.append((best_pos_global[0], best_pos_global[1], best_pos_global[2], best_orient_global[0], best_orient_global[1], best_orient_global[2]))
                result[idx] = (best_pos_global[0], best_pos_global[1], best_pos_global[2], best_orient_global[0], best_orient_global[1], best_orient_global[2])
        
        return result, placed

    def compute_fill(result):
        total = 0
        for idx, (x, y, z, w, h, d) in result.items():
            total += w * h * d
        return total / container_vol

    # Sorting Strategies
    def sort_by_volume(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
        
    def sort_by_min_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)

    best_result = {}
    best_fill = 0.0

    # Strategies: (order_generator, use_best_fit)
    # We try First Fit (fast) and Best Fit (slower, potentially better)
    strategies = [
        (sort_by_volume(boxes), False),
        (sort_by_volume(boxes), True),
        (sort_by_max_dim(boxes), False),
        (sort_by_max_dim(boxes), True),
        (sort_by_min_dim(boxes), False),
        (sort_by_min_dim(boxes), True),
    ]
    
    # Add some random restarts
    rng = random.Random(42)
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        random_order = list(range(n))
        rng.shuffle(random_order)
        strategies.append((random_order, False))
        strategies.append((random_order, True))

    for order, use_best_fit in strategies:
        if time.time() > DEADLINE:
            break
            
        result, placed = solve_instance(boxes, container, order, use_best_fit)
        fill = compute_fill(result)
        
        if fill > best_fill:
            best_fill = fill
            best_result = result

    # --- Local Search / Polish Phase ---
    # Try to improve the best result by moving boxes or inserting remaining ones.
    if best_result and time.time() < DEADLINE - 10:
        # Reconstruct placed list from best_result
        placed_list = []
        for idx, pos in best_result.items():
            placed_list.append(pos)
            
        # 1. Try to insert unplaced boxes
        unplaced = [i for i in range(n) if i not in best_result]
        # Sort unplaced by volume descending
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in unplaced:
            if time.time() > DEADLINE:
                break
            w, h, d = boxes[idx]
            rotations = get_rotations(w, h, d)
            
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            for ow, oh, od in rotations:
                if ow > W or oh > H or od > D:
                    continue
                candidates = get_grid_candidates(placed_list, ow, oh, od)
                for (tx, ty, tz) in candidates:
                    if can_place(tx, ty, tz, ow, oh, od, placed_list):
                        score = tx + ty + tz
                        if score < best_score:
                            best_score = score
                            best_pos = (tx, ty, tz)
                            best_orient = (ow, oh, od)
                            break # First fit for this rotation
            
            if best_pos:
                best_result[idx] = best_pos + best_orient
                placed_list.append(best_pos + best_orient)
                best_fill = compute_fill(best_result)

        # 2. Local Search: Try to move existing boxes to create space or improve packing
        # This is expensive. We'll do a limited number of iterations.
        improved = True
        max_ls_iterations = 5
        ls_iter = 0
        
        while improved and ls_iter < max_ls_iterations and time.time() < DEADLINE - 5:
            improved = False
            ls_iter += 1
            
            # Update placed_list from best_result to ensure consistency
            placed_list = list(best_result.values())
            
            # Try to move each box
            # To save time, only try to move boxes that are not at (0,0,0) or have "waste" above them?
            # For simplicity, try all, but limit the candidate search.
            
            for idx in list(best_result.keys()):
                if time.time() > DEADLINE:
                    break
                    
                x, y, z, w, h, d = best_result[idx]
                # Remove this box from placed_list for collision check
                other_placed = [p for i, p in enumerate(best_result.values()) if i != idx]
                
                rotations = get_rotations(w, h, d)
                best_new_pos = None
                best_new_orient = None
                best_score = x + y + z
                
                # Only try to improve if we can find a spot with lower x+y+z
                # Or if we can free up space for others? 
                # Simple heuristic: move to lower x+y+z.
                
                for ow, oh, od in rotations:
                    if ow > W or oh > H or od > D:
                        continue
                    
                    candidates = get_grid_candidates(other_placed, ow, oh, od)
                    # Limit candidates to those with score < current best_score
                    for (tx, ty, tz) in candidates:
                        score = tx + ty + tz
                        if score >= best_score:
                            continue # We only want to improve (lower score)
                        
                        if can_place(tx, ty, tz, ow, oh, od, other_placed):
                            best_score = score
                            best_new_pos = (tx, ty, tz)
                            best_new_orient = (ow, oh, od)
                            break # First fit for this rotation
                
                if best_new_pos is not None and (best_new_pos != (x, y, z) or best_new_orient != (w, h, d)):
                    best_result[idx] = best_new_pos + best_new_orient
                    improved = True
                    
                    # After moving, try to fit unplaced boxes again?
                    # This is complex. Let's just do one pass of moving.
                    
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
