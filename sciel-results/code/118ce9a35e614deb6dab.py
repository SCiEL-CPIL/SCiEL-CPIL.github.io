import time
import random
import itertools
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # 1. Generate candidate configurations
    # Strategy: Explore structured lattices that are known to produce many small-area triangles.
    # The seed is a 6x5 grid. We will explore:
    # - Other rectangular grids (3x10, 2x15, etc.)
    # - "Sheared" grids (parallelograms)
    # - "Staggered" grids (brick wall)
    # - Randomized local search with incremental updates for the best target area.
    
    candidates = []
    
    # Standard grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30 and len(set(pts)) == 30:
            # Center them to be in [-30, 30]
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            candidates.append(pts)
            
    # Sheared grids (parallelograms)
    # Points: (x + k*y, y) for x in range(w), y in range(h)
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3)]:
        for k in [1, -1, 2, -2]:
            pts = []
            valid = True
            for x in range(w):
                for y in range(h):
                    px = x + k * y
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid: break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                # Center
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

    # Staggered grids (brick wall)
    # Row y: x in range(offset, offset + w)
    for w, h in [(6, 5), (5, 6)]:
        for offset_pattern in [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0]]:
            if len(offset_pattern) != h: continue
            pts = []
            valid = True
            for y in range(h):
                off = offset_pattern[y]
                for x in range(w):
                    px = x + off
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid: break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

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
        return [(0,0)] * 30 # Fallback, though shouldn't happen

    # 2. Local Search with Incremental Updates
    # We focus on improving the count of the current best area.
    # Moving a point P to P' changes the triangle count for area A by:
    # - Removing triangles involving P and area A
    # - Adding triangles involving P' and area A
    # This is O(N^2) per move evaluation instead of O(N^3).
    
    current_points = list(best_points)
    current_score = best_score
    current_area = best_area
    
    # Precompute area counts for current state to help? 
    # Actually, for incremental updates, we just need to check if the move increases the count of current_area.
    # But the "best area" might change. So we should periodically re-evaluate the full max.
    
    # Let's run a hill-climbing loop.
    # We will try to move each point to a nearby integer coordinate.
    # If the count of current_area increases, we keep it.
    # If the total score (max multiplicity) increases, we update best.
    
    # To be safe and fast, we will:
    # 1. Pick a random point.
    # 2. Try all valid integer moves in a small neighborhood (e.g., Chebyshev distance 2).
    # 3. For each move, calculate the delta in count for current_area.
    # 4. If delta > 0, perform the move, update current_points, and update current_score (approx).
    # 5. Every K steps or if time is running out, do a full re-evaluation to find the true max area and score.
    
    # Helper: Calculate delta in count of area A when moving point i from old_pos to new_pos
    def calc_delta_area(points, idx, old_pos, new_pos, target_area):
        delta = 0
        x1, y1 = new_pos
        # For every pair (j, k) not involving i:
        # Triangle (new_pos, Pj, Pk) has area A?
        # We need to check all pairs (j, k) where j, k != idx.
        # There are ~29*28/2 = 406 pairs.
        # This is fast enough.
        
        for j in range(len(points)):
            if j == idx: continue
            x2, y2 = points[j]
            for k in range(j + 1, len(points)):
                if k == idx: continue
                x3, y3 = points[k]
                
                # Area of (new_pos, Pj, Pk)
                det_new = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det_new == target_area:
                    delta += 1
                
                # Area of (old_pos, Pj, Pk)
                ox, oy = old_pos
                det_old = abs((x2 - ox) * (y3 - oy) - (y2 - oy) * (x3 - ox))
                if det_old == target_area:
                    delta -= 1
        return delta

    # Main loop
    steps = 0
    while time.time() < deadline - 2:
        steps += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Try moves in a 5x5 neighborhood around old position
        best_move = None
        best_delta = 0
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0: continue
                new_x = old_x + dx
                new_y = old_y + dy
                
                # Bounds check
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                # Distinctness check
                new_pos = (new_x, new_y)
                if new_pos in current_points:
                    continue
                
                # Calculate delta for current_target_area
                delta = calc_delta_area(current_points, idx, (old_x, old_y), new_pos, current_area)
                
                if delta > best_delta:
                    best_delta = delta
                    best_move = new_pos
        
        if best_move:
            # Apply move
            current_points[idx] = best_move
            # Update current_score approximately? 
            # We don't know the exact new max multiplicity, but we know the count of current_area increased.
            # It's possible another area became larger. 
            # To be safe, we update the score only if we re-evaluate.
            # But we can track the count of current_area.
            # Let's just mark that we need a full check soon.
            pass
        else:
            # No improvement for current_area. 
            # Maybe try a different area? Or just continue.
            # To avoid getting stuck, let's occasionally re-evaluate the full configuration 
            # to see if the "best area" has shifted.
            if steps % 100 == 0:
                score, area = _calculate_max_multiplicity(current_points)
                if score > current_score:
                    current_score = score
                    current_area = area
                elif score == current_score and area != current_area:
                    # Tie, maybe switch to this area to try to improve it?
                    # Or keep the one with higher potential? 
                    # Let's keep the current one unless the score drops.
                    pass

    # Final full evaluation
    final_score, final_area = _calculate_max_multiplicity(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points
    else:
        # If local search didn't improve the global best, return the global best found in candidates
        # But wait, if current_points is a modification of best_points, and it didn't improve, 
        # it might be worse. So we should compare.
        # Actually, we started with best_points from candidates. 
        # If local search made it worse, we should return the original best_points.
        # Let's check if current_points is better than the stored best_points.
        # We already have best_score from candidates.
        # If final_score < best_score, return best_points.
        # If final_score >= best_score, return current_points.
        if final_score < best_score:
            return best_points
        else:
            return current_points

    return current_points

def _calculate_max_multiplicity(points):
    """Calculate the maximum multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0, 0
        
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
                    
    if not areas:
        return 0, 0
        
    max_mult = max(areas.values())
    max_area = [a for a, m in areas.items() if m == max_mult][0]
    return max_mult, max_area
# EVOLVE-BLOCK-END
