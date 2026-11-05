import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 30
    candidates = []
    
    # Generate diverse initial candidates
    # Standard grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30 and len(set(pts)) == 30:
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            candidates.append(pts)
            
    # Sheared grids
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
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

    # Staggered grids
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
    
    # Random configurations
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        pts = []
        attempts = 0
        while len(pts) < 30 and attempts < 500:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
            attempts += 1
        if len(pts) == 30:
            candidates.append(pts)

    if not candidates:
        return [(0, 0)] * 30

    # Evaluate all candidates
    best_score = 0
    best_points = candidates[0]
    best_areas = {}
    
    for pts in candidates:
        score, areas = _calculate_all_areas(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            best_areas = areas
            
    # Multi-target local search
    # Get top 5 areas to optimize simultaneously
    top_areas = sorted(best_areas.items(), key=lambda x: -x[1])[:5]
    target_areas = [a for a, m in top_areas]
    target_counts = {a: m for a, m in top_areas}
    
    current_points = list(best_points)
    current_score = best_score
    
    # Precompute for vectorized delta calculation
    n = 30
    
    def get_top_areas(points, k=5):
        """Get top-k areas and their counts"""
        areas = _calculate_all_areas(points)[1]
        if not areas:
            return [], {}
        sorted_areas = sorted(areas.items(), key=lambda x: -x[1])[:k]
        return [a for a, m in sorted_areas], {a: m for a, m in sorted_areas}
    
    def calc_vectorized_deltas(points, idx, old_pos, new_pos, targets):
        """
        Vectorized calculation of delta for each target area.
        Returns list of (target_area, delta) pairs.
        """
        x1, y1 = new_pos
        ox, oy = old_pos
        
        # Get all other points
        others = [points[j] for j in range(n) if j != idx]
        
        if len(others) < 2:
            return {a: 0 for a in targets}
        
        # For each pair of other points, compute area with new_pos and old_pos
        # Vectorize over pairs
        results = {a: 0 for a in targets}
        
        # Build arrays of other points
        other_arr = np.array(others, dtype=np.float64)  # shape (29, 2)
        
        # Create all pairs (i, j) with i < j
        pair_indices = []
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                pair_indices.append((i, j))
        
        if not pair_indices:
            return results
        
        pair_indices = np.array(pair_indices)
        p1 = other_arr[pair_indices[:, 0]]  # shape (num_pairs, 2)
        p2 = other_arr[pair_indices[:, 1]]  # shape (num_pairs, 2)
        
        # Compute determinants for new position
        # det = |(x2-x1)(y3-y1) - (y2-y1)(x3-x1)|
        # For new_pos as first point:
        dx1_new = p1[:, 0] - x1
        dy1_new = p1[:, 1] - y1
        dx2_new = p2[:, 0] - x1
        dy2_new = p2[:, 1] - y1
        det_new = np.abs(dx1_new * dy2_new - dy1_new * dx2_new)
        
        # For old position:
        dx1_old = p1[:, 0] - ox
        dy1_old = p1[:, 1] - oy
        dx2_old = p2[:, 0] - ox
        dy2_old = p2[:, 1] - oy
        det_old = np.abs(dx1_old * dy2_old - dy1_old * dx2_old)
        
        # Count for each target
        for a in targets:
            count_new = np.sum(det_new == a)
            count_old = np.sum(det_old == a)
            results[a] = count_new - count_old
        
        return results
    
    # Main local search loop with multi-target optimization
    no_improve_count = 0
    max_no_improve = 200
    steps = 0
    
    while time.time() < deadline - 3:
        steps += 1
        
        # Occasionally re-evaluate and update targets
        if steps % 500 == 0:
            score, areas = _calculate_all_areas(current_points)
            if score > current_score:
                current_score = score
            top_areas = sorted(areas.items(), key=lambda x: -x[1])[:5]
            target_areas = [a for a, m in top_areas]
            target_counts = {a: m for a, m in top_areas}
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Try moves in a neighborhood
        best_move = None
        best_combined_delta = 0
        
        # Use weighted sum of deltas across target areas
        # Weight by current count to prioritize high-count areas
        weights = {}
        for a in target_areas:
            weights[a] = target_counts.get(a, 0)
        
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0: continue
                new_x = old_x + dx
                new_y = old_y + dy
                
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_pos = (new_x, new_y)
                if new_pos in current_points:
                    continue
                
                deltas = calc_vectorized_deltas(current_points, idx, (old_x, old_y), new_pos, target_areas)
                
                # Combined score: weighted sum of positive deltas
                combined = 0
                for a in target_areas:
                    d = deltas[a]
                    if d > 0:
                        combined += d * weights[a]
                    elif d < 0:
                        # Penalize losing count in high-count areas more
                        combined += d * weights[a] * 1.5
                
                if combined > best_combined_delta:
                    best_combined_delta = combined
                    best_move = new_pos
        
        if best_move and best_combined_delta > 0:
            current_points[idx] = best_move
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Re-evaluate and potentially reset
                score, areas = _calculate_all_areas(current_points)
                if score > current_score:
                    current_score = score
                    best_points = list(current_points)
                    top_areas = sorted(areas.items(), key=lambda x: -x[1])[:5]
                    target_areas = [a for a, m in top_areas]
                    target_counts = {a: m for a, m in top_areas}
                    no_improve_count = 0
                else:
                    # Reset to best known
                    current_points = list(best_points)
                    no_improve_count = 0
                
                # Try bigger jumps
                for _ in range(30):
                    if time.time() > deadline - 3:
                        break
                    idx2 = random.randint(0, 29)
                    old_x2, old_y2 = current_points[idx2]
                    for _2 in range(10):
                        new_x2 = random.randint(-30, 30)
                        new_y2 = random.randint(-30, 30)
                        new_pos2 = (new_x2, new_y2)
                        if new_pos2 in current_points:
                            continue
                        deltas = calc_vectorized_deltas(current_points, idx2, (old_x2, old_y2), new_pos2, target_areas)
                        combined = sum(max(0, deltas[a]) * weights[a] for a in target_areas)
                        if combined > 0:
                            current_points[idx2] = new_pos2
                            no_improve_count = 0
                            break

    # Final full evaluation
    final_score, final_areas = _calculate_all_areas(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points

    return best_points

def _calculate_all_areas(points):
    """Calculate all areas and their multiplicities."""
    n = len(points)
    if n < 3:
        return 0, {}
        
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
        return 0, {}
        
    max_mult = max(areas.values())
    return max_mult, areas

def _calculate_max_multiplicity(points):
    """Compatibility wrapper for the original interface."""
    score, areas = _calculate_all_areas(points)
    if not areas:
        return 0, 0
    max_area = max(areas, key=lambda a: areas[a])
    return score, max_area
