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
    candidates = []
    
    # Standard grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30 and len(set(pts)) == 30:
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            candidates.append(pts)
            
    # Sheared grids (parallelograms)
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3)]:
        for k in [1, -1, 2, -2, 3, -3]:
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
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

    # Staggered grids (brick wall)
    for w, h in [(6, 5), (5, 6)]:
        for offset_pattern in [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]]:
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
                
    # Add some random points to diversify initial candidates
    for _ in range(20):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-15, 15)
            y = random.randint(-15, 15)
            pts.add((x, y))
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

    # 2. Local Search with Global Max Awareness
    # We track the top K most frequent areas.
    # A move is accepted if it increases the count of the current max area,
    # or if it increases the count of a sub-max area enough to become the new max.
    
    current_points = list(best_points)
    current_score = best_score
    current_area = best_area
    
    # Helper: Calculate delta in count of area A when moving point i from old_pos to new_pos
    def calc_delta_area(points, idx, old_pos, new_pos, target_area):
        delta = 0
        x1, y1 = new_pos
        for j in range(len(points)):
            if j == idx: continue
            x2, y2 = points[j]
            for k in range(j + 1, len(points)):
                if k == idx: continue
                x3, y3 = points[k]
                
                det_new = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det_new == target_area:
                    delta += 1
                
                ox, oy = old_pos
                det_old = abs((x2 - ox) * (y3 - oy) - (y2 - oy) * (x3 - ox))
                if det_old == target_area:
                    delta -= 1
        return delta

    # Precompute top areas for the initial best configuration
    def get_top_areas(points, k=3):
        areas = {}
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for kk in range(j + 1, n):
                    x3, y3 = points[kk]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return []
        sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
        return sorted_areas[:k]

    top_areas = get_top_areas(current_points, 3)
    # top_areas is a list of (area, count)
    
    steps = 0
    while time.time() < deadline - 2:
        steps += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Try moves in a 5x5 neighborhood around old position
        best_move = None
        best_eval_score = -1
        best_eval_area = -1
        
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
                
                # Evaluate move against top areas
                move_best_score = -1
                move_best_area = -1
                
                for area, count in top_areas:
                    delta = calc_delta_area(current_points, idx, (old_x, old_y), new_pos, area)
                    new_count = count + delta
                    if new_count > move_best_score:
                        move_best_score = new_count
                        move_best_area = area
                        
                if move_best_score > best_eval_score:
                    best_eval_score = move_best_score
                    best_eval_area = move_best_area
                    best_move = new_pos

        if best_move and best_eval_score > current_score:
            # Apply move
            current_points[idx] = best_move
            current_score = best_eval_score
            current_area = best_eval_area
            
            # Update top_areas periodically
            if steps % 50 == 0:
                top_areas = get_top_areas(current_points, 3)
                
    # Final full evaluation
    final_score, final_area = _calculate_max_multiplicity(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points
    else:
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
