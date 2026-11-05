import time
import random
import itertools
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s safety margin for 90s limit

    # Precompute all triangle index triples
    N = 30
    triples = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                triples.append((i, j, k))
    n_triples = len(triples)

    def _calculate_max_multiplicity(points):
        """Calculate the maximum multiplicity of any positive doubled area."""
        areas = {}
        for i, j, k in triples:
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det > 0:
                areas[det] = areas.get(det, 0) + 1

        if not areas:
            return 0, 0

        max_mult = max(areas.values())
        max_area = [a for a, m in areas.items() if m == max_mult][0]
        return max_mult, max_area

    def _calc_delta(points, idx, old_pos, new_pos, target_area):
        """
        Calculate the change in the count of triangles with 'target_area' 
        when moving point idx from old_pos to new_pos.
        Only considers triangles that include point idx.
        """
        delta = 0
        x1, y1 = new_pos
        ox, oy = old_pos

        # Precompute coordinates of all other points
        other_indices = [i for i in range(N) if i != idx]
        other_coords = [(points[i][0], points[i][1]) for i in other_indices]
        n_other = len(other_coords)

        for a in range(n_other):
            x2, y2 = other_coords[a]
            for b in range(a + 1, n_other):
                x3, y3 = other_coords[b]

                det_new = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det_new == target_area:
                    delta += 1

                det_old = abs((x2 - ox) * (y3 - oy) - (y2 - oy) * (x3 - ox))
                if det_old == target_area:
                    delta -= 1

        return delta

    # 1. Generate diverse initial candidates
    candidates = []
    
    # A. Rectangular Grids
    for w in range(1, 31):
        for h in range(1, 31):
            if w * h != 30: continue
            # Check if it fits in bounds (roughly, we center later)
            # Max coordinate diff is max(w-1, h-1) + shear. 
            # For grid, max diff is w-1 and h-1.
            # Centered, range is [-w//2, w//2] etc.
            # w=30, h=1 -> x in [-15, 15]. OK.
            # w=1, h=30 -> y in [-15, 15]. OK.
            pts = [(x, y) for x in range(w) for y in range(h)]
            if len(pts) == 30 and len(set(pts)) == 30:
                cx, cy = w // 2, h // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)
                
    # B. Sheared Grids (Parallelograms)
    # Points: (x + k*y, y)
    for w in range(1, 31):
        for h in range(1, 31):
            if w * h != 30: continue
            for k in [-2, -1, 0, 1, 2]:
                pts = []
                valid = True
                for x in range(w):
                    for y in range(h):
                        px = x + k * y
                        py = y
                        pts.append((px, py))
                # Check bounds and distinctness
                if len(set(pts)) != 30:
                    valid = False
                else:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    # Center
                    cx = (min(xs) + max(xs)) // 2
                    cy = (min(ys) + max(ys)) // 2
                    pts_c = [(x - cx, y - cy) for x, y in pts]
                    # Check if centered points are in [-30, 30]
                    if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
                        candidates.append(pts_c)
                    else:
                        valid = False
                        
    # C. "Staggered" or Brick Wall Patterns
    # Try a few specific patterns for 6x5 and 5x6
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10)]:
        # Simple zigzag
        pts = []
        for y in range(h):
            off = y % 2
            for x in range(w):
                px = x + off
                py = y
                pts.append((px, py))
        if len(set(pts)) == 30:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            pts_c = [(x - cx, y - cy) for x, y in pts]
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
                candidates.append(pts_c)

    # D. Random Points
    for _ in range(50):
        pts = set()
        while len(pts) < 30:
            pts.add((random.randint(-15, 15), random.randint(-15, 15)))
        candidates.append(list(pts))

    # Evaluate all candidates
    best_score = 0
    best_points = candidates[0] if candidates else [(0,0)]
    best_area = 1
    
    for pts in candidates:
        score, area = _calculate_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            best_area = area

    if not candidates:
        return [(0,0)] * 30

    # 2. Local Search
    # We will run multiple rounds of local search starting from the best candidate.
    # We also allow "restarts" from other good candidates if time permits.
    
    # Sort candidates by score descending to prioritize good starts
    scored_candidates = []
    for pts in candidates:
        score, area = _calculate_max_multiplicity(pts)
        scored_candidates.append((score, area, pts))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    current_best_score = best_score
    current_best_points = best_points
    current_best_area = best_area

    # We will iterate through the top candidates and try to improve them.
    # However, to save time, we will primarily focus on the best one,
    # and maybe the second best if time allows.
    
    for start_score, start_area, start_pts in scored_candidates[:5]:
        if time.time() > deadline - 5:
            break
            
        current_points = list(start_pts)
        current_area = start_area
        current_score = start_score
        
        # If this start is worse than our global best, we might still try to improve it
        # to see if it can beat the global best.
        
        # Local search loop
        no_improve_count = 0
        max_no_improve = 200 # Reset or break if stuck
        
        while time.time() < deadline - 2:
            if no_improve_count > max_no_improve:
                # Try to jump to a random area or just break to next start
                # Let's break to next start candidate
                break
                
            # Pick a random point to move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            
            # Try moves in a 5x5 neighborhood
            best_move = None
            best_delta = 0
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0: continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    
                    new_pos = (new_x, new_y)
                    if new_pos in current_points:
                        continue
                    
                    delta = _calc_delta(current_points, idx, (old_x, old_y), new_pos, current_area)
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_move = new_pos
            
            if best_move:
                # Apply move
                current_points[idx] = best_move
                no_improve_count = 0
                # We don't update current_score immediately because max area might change.
                # We update it periodically.
            else:
                no_improve_count += 1
                # If stuck, maybe re-evaluate full score to see if we found a better area
                # or if we should switch target area.
                if no_improve_count % 50 == 0:
                    score, area = _calculate_max_multiplicity(current_points)
                    if score > current_score:
                        current_score = score
                        current_area = area
                        no_improve_count = 0 # Reset as we found an improvement in global score
                    elif score == current_score and area != current_area:
                        # Switch target to this area?
                        # Maybe, if it has higher potential? Hard to know.
                        # Let's stick to the one that gave the score.
                        pass

        # After local search on this start, check if it beat the global best
        final_score, final_area = _calculate_max_multiplicity(current_points)
        if final_score > current_best_score:
            current_best_score = final_score
            current_best_points = current_points
            current_best_area = final_area
            # If we found a new best, we might want to continue local search on it?
            # But we are in a loop over starts.
            # Let's just break if we have a very high score?
            # Or just continue to next start to see if another path is better.
            # Given time constraints, breaking early if we have a strong score is good.
            # But let's allow the loop to finish the top 5 starts.

    # Final check
    if current_best_score > best_score:
        return current_best_points
    else:
        return best_points

# EVOLVE-BLOCK-END
